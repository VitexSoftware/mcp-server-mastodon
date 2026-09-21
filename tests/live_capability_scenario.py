#!/usr/bin/env python3
"""Live capability scenario for Mastodon MCP Server.

Exercises every MCP tool against a real Mastodon instance and reports whether
each capability returns usable data (or correctly refuses writes under
READ_ONLY).

Usage:
  MASTODON_INSTANCE=https://f.cz \\
  MASTODON_ACCESS_TOKEN=... \\
  READ_ONLY=true \\
    python tests/live_capability_scenario.py --json-out /tmp/mastodon-live.json

  python tests/live_capability_scenario.py \\
    --instance https://f.cz \\
    --access-token "$MASTODON_ACCESS_TOKEN" \\
    --json-out /tmp/mastodon-live.json

Exit code is 0 only when every non-skipped check passes.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@dataclass
class CheckResult:
    name: str
    kind: str  # tool | meta | guard
    ok: bool
    detail: str = ""
    sample: Any = None
    skipped: bool = False


@dataclass
class ScenarioReport:
    instance: str
    results: List[CheckResult] = field(default_factory=list)

    def add(self, result: CheckResult) -> None:
        self.results.append(result)

    @property
    def passed(self) -> int:
        return sum(1 for r in self.results if r.ok and not r.skipped)

    @property
    def failed(self) -> int:
        return sum(1 for r in self.results if not r.ok and not r.skipped)

    @property
    def skipped(self) -> int:
        return sum(1 for r in self.results if r.skipped)


def _parse_payload(data: Any) -> Any:
    if isinstance(data, str):
        try:
            return json.loads(data)
        except (json.JSONDecodeError, TypeError):
            return data
    return data


def _is_error_payload(data: Any) -> Optional[str]:
    data = _parse_payload(data)
    if isinstance(data, str):
        low = data.lower()
        if "error" in low or "traceback" in low or "exception" in low:
            return data[:300]
        return None
    if not isinstance(data, dict):
        return None
    if data.get("error") is True or data.get("success") is False:
        return str(data.get("message") or data.get("reason") or data.get("error") or data)[:300]
    if "error" in data and data["error"] not in (None, False, ""):
        return str(data["error"])[:300]
    return None


def _has_usable_data(data: Any) -> Tuple[bool, str]:
    data = _parse_payload(data)
    err = _is_error_payload(data)
    if err:
        return False, err
    if data is None:
        return False, "null response"
    if isinstance(data, str):
        return (bool(data.strip()), "empty string" if not data.strip() else "ok string")
    if isinstance(data, list):
        return (True, f"list len={len(data)}")
    if isinstance(data, dict):
        if data.get("success") is True:
            return True, "success=True"
        if data.get("id") is not None:
            return True, f"id={data.get('id')}"
        if any(k in data for k in ("uri", "url", "username", "acct", "title", "version", "domain")):
            return True, f"keys={list(data.keys())[:8]}"
        if any(k in data for k in ("accounts", "statuses", "hashtags", "ancestors", "descendants")):
            return True, f"search/context keys={list(data.keys())[:8]}"
        if data:
            return True, f"keys={list(data.keys())[:8]}"
        return False, "empty object"
    return True, f"type={type(data).__name__}"


def _tool_is_readonly(tool: Any) -> bool:
    ann = getattr(tool, "annotations", None)
    if ann is None:
        return True
    if hasattr(ann, "read_only_hint"):
        return bool(ann.read_only_hint)
    if hasattr(ann, "readOnlyHint"):
        return bool(ann.readOnlyHint)
    if isinstance(ann, dict):
        return bool(ann.get("read_only_hint", ann.get("readOnlyHint", True)))
    return True


def _first_id(payload: Any, *keys: str) -> Optional[str]:
    data = _parse_payload(payload)
    if isinstance(data, list) and data:
        first = data[0]
        if isinstance(first, dict):
            for key in keys or ("id",):
                if first.get(key) is not None:
                    return str(first[key])
            if first.get("id") is not None:
                return str(first["id"])
    if isinstance(data, dict):
        for key in keys or ("id",):
            if data.get(key) is not None:
                return str(data[key])
        # nested lists common in search results
        for nest in ("statuses", "accounts", "hashtags"):
            items = data.get(nest)
            if isinstance(items, list) and items and isinstance(items[0], dict):
                if items[0].get("id") is not None:
                    return str(items[0]["id"])
                if nest == "hashtags" and items[0].get("name"):
                    return str(items[0]["name"])
    return None


def _run_check(
    report: ScenarioReport,
    name: str,
    kind: str,
    fn: Callable[[], Any],
    *,
    require_data: bool = True,
    skip_reason: Optional[str] = None,
) -> Optional[Any]:
    if skip_reason:
        report.add(CheckResult(name=name, kind=kind, ok=True, detail=skip_reason, skipped=True))
        return None
    try:
        raw = fn()
        ok, detail = _has_usable_data(raw) if require_data else (True, "invoked")
        sample = _parse_payload(raw)
        try:
            preview = json.dumps(sample, ensure_ascii=False, default=str)
        except TypeError:
            preview = str(sample)
        report.add(
            CheckResult(
                name=name,
                kind=kind,
                ok=ok,
                detail=detail,
                sample=preview[:800],
            )
        )
        return raw
    except Exception as exc:  # noqa: BLE001 — scenario must keep going
        report.add(
            CheckResult(
                name=name,
                kind=kind,
                ok=False,
                detail=f"{type(exc).__name__}: {exc}",
                sample=traceback.format_exc()[-600:],
            )
        )
        return None


def _assert_write_blocked(report: ScenarioReport, name: str, fn: Callable[[], Any]) -> None:
    try:
        raw = fn()
        payload = _parse_payload(raw)
        refused = False
        if isinstance(payload, dict) and (
            "read-only" in str(payload).lower() or payload.get("success") is False
        ):
            refused = True
        elif isinstance(payload, str) and "read-only" in payload.lower():
            refused = True
        report.add(
            CheckResult(
                name=f"readonly_guard:{name}",
                kind="guard",
                ok=refused,
                detail="write returned without raising" if not refused else "refused in payload",
                sample=str(payload)[:300],
            )
        )
    except ValueError as exc:
        ok = "read-only" in str(exc).lower()
        report.add(
            CheckResult(
                name=f"readonly_guard:{name}",
                kind="guard",
                ok=ok,
                detail=str(exc),
            )
        )
    except Exception as exc:  # noqa: BLE001
        report.add(
            CheckResult(
                name=f"readonly_guard:{name}",
                kind="guard",
                ok=False,
                detail=f"unexpected {type(exc).__name__}: {exc}",
            )
        )


def run_scenario(
    instance: str,
    access_token: str,
    *,
    read_only: bool = True,
) -> ScenarioReport:
    os.environ["MASTODON_INSTANCE"] = instance
    os.environ["MASTODON_ACCESS_TOKEN"] = access_token
    os.environ["READ_ONLY"] = "true" if read_only else "false"

    import importlib
    import mastodon_mcp_server.server as server

    # Reset cached client so env takes effect even on re-runs
    server._client = None
    server = importlib.reload(server)
    server._client = None

    report = ScenarioReport(instance=instance)
    tools = asyncio.run(server.mcp.list_tools())
    tool_by_name = {t.name: t for t in tools}
    fn_by_name = {
        name: getattr(server, name)
        for name in tool_by_name
        if callable(getattr(server, name, None))
    }

    report.add(
        CheckResult(
            name="tool_catalog",
            kind="meta",
            ok=len(tools) > 0,
            detail=f"{len(tools)} tools registered",
        )
    )

    # Annotation sanity: every write tool must declare readOnlyHint=False and call validate_write
    for t in tools:
        is_ro = _tool_is_readonly(t)
        fn = fn_by_name.get(t.name)
        if fn is None:
            report.add(
                CheckResult(
                    name=f"callable:{t.name}",
                    kind="meta",
                    ok=False,
                    detail="registered in MCP but missing Python callable",
                )
            )
            continue
        # Heuristic: tools that call validate_write should not be annotated read-only
        src = ""
        try:
            import inspect

            src = inspect.getsource(fn)
        except (OSError, TypeError):
            src = ""
        has_validate = "validate_write()" in src
        if has_validate and is_ro:
            report.add(
                CheckResult(
                    name=f"annotation:{t.name}",
                    kind="meta",
                    ok=False,
                    detail="calls validate_write but annotated readOnlyHint=True",
                )
            )
        elif (not has_validate) and (not is_ro) and t.name not in (
            # write annotations without validate would be a bug — catch below via guards
        ):
            # write-annotated tools without validate_write are suspicious
            if "validate_write" not in src:
                report.add(
                    CheckResult(
                        name=f"annotation:{t.name}",
                        kind="meta",
                        ok=False,
                        detail="annotated write (readOnlyHint=False) but no validate_write()",
                    )
                )

    # ---- instance / identity ----
    _run_check(report, "instance_info", "tool", server.instance_info)
    me = _run_check(report, "account_verify", "tool", server.account_verify)
    me_id = _first_id(me)

    # ---- account reads ----
    _run_check(
        report,
        "account_get",
        "tool",
        lambda: server.account_get(account_id=me_id),
        skip_reason=None if me_id else "no authenticated account id",
    )
    search_acc = _run_check(
        report,
        "account_search",
        "tool",
        lambda: server.account_search(query="mastodon", limit=5),
    )
    other_id = _first_id(search_acc)
    if other_id == me_id and search_acc is not None:
        parsed = _parse_payload(search_acc)
        if isinstance(parsed, list):
            for item in parsed[1:]:
                if isinstance(item, dict) and item.get("id") is not None:
                    other_id = str(item["id"])
                    break

    statuses = _run_check(
        report,
        "account_statuses",
        "tool",
        lambda: server.account_statuses(account_id=me_id, limit=5),
        skip_reason=None if me_id else "no authenticated account id",
    )
    status_id = _first_id(statuses)

    _run_check(
        report,
        "account_followers",
        "tool",
        lambda: server.account_followers(account_id=me_id, limit=5),
        skip_reason=None if me_id else "no authenticated account id",
    )
    _run_check(
        report,
        "account_following",
        "tool",
        lambda: server.account_following(account_id=me_id, limit=5),
        skip_reason=None if me_id else "no authenticated account id",
    )
    rel_ids = [x for x in (me_id, other_id) if x]
    _run_check(
        report,
        "account_relationships",
        "tool",
        lambda: server.account_relationships(account_ids=rel_ids[:1] or [me_id]),
        skip_reason=None if me_id else "no account id for relationships",
    )

    # ---- timelines ----
    home = _run_check(report, "timeline_home", "tool", lambda: server.timeline_home(limit=5))
    if status_id is None:
        status_id = _first_id(home)
    local = _run_check(report, "timeline_local", "tool", lambda: server.timeline_local(limit=5))
    if status_id is None:
        status_id = _first_id(local)
    pub = _run_check(report, "timeline_public", "tool", lambda: server.timeline_public(limit=5))
    if status_id is None:
        status_id = _first_id(pub)

    tag_src = _parse_payload(local) or _parse_payload(pub) or []
    hashtag = "mastodon"
    if isinstance(tag_src, list):
        for st in tag_src:
            if isinstance(st, dict) and st.get("tags"):
                tags = st["tags"]
                if isinstance(tags, list) and tags and isinstance(tags[0], dict) and tags[0].get("name"):
                    hashtag = str(tags[0]["name"])
                    break
    _run_check(
        report,
        "timeline_hashtag",
        "tool",
        lambda: server.timeline_hashtag(hashtag=hashtag, limit=5),
    )

    # ---- statuses ----
    _run_check(
        report,
        "status_get",
        "tool",
        lambda: server.status_get(status_id=status_id),
        skip_reason=None if status_id else "no status id from timelines/account",
    )
    _run_check(
        report,
        "status_context",
        "tool",
        lambda: server.status_context(status_id=status_id),
        skip_reason=None if status_id else "no status id",
    )
    _run_check(
        report,
        "status_favourited_by",
        "tool",
        lambda: server.status_favourited_by(status_id=status_id),
        skip_reason=None if status_id else "no status id",
    )
    _run_check(
        report,
        "status_reblogged_by",
        "tool",
        lambda: server.status_reblogged_by(status_id=status_id),
        skip_reason=None if status_id else "no status id",
    )

    # ---- notifications / search / trending ----
    _run_check(report, "notifications_get", "tool", lambda: server.notifications_get(limit=5))
    search_res = _run_check(
        report,
        "search",
        "tool",
        lambda: server.search(query="mastodon", limit=5),
    )
    _ = search_res
    _run_check(report, "trending_tags", "tool", lambda: server.trending_tags(limit=5))
    _run_check(report, "trending_statuses", "tool", lambda: server.trending_statuses(limit=5))
    _run_check(report, "trending_links", "tool", lambda: server.trending_links(limit=5))

    # ---- favourites / bookmarks / lists ----
    _run_check(report, "favourites", "tool", lambda: server.favourites(limit=5))
    _run_check(report, "bookmarks", "tool", lambda: server.bookmarks(limit=5))
    lists = _run_check(report, "lists_get", "tool", server.lists_get)
    list_id = _first_id(lists)
    _run_check(
        report,
        "list_accounts",
        "tool",
        lambda: server.list_accounts(list_id=list_id, limit=5),
        skip_reason=None if list_id else "no lists on this account",
    )

    # ---- follow requests / mutes / blocks / directory ----
    _run_check(report, "follow_requests", "tool", lambda: server.follow_requests(limit=5))
    _run_check(report, "mutes", "tool", lambda: server.mutes(limit=5))
    _run_check(report, "blocks", "tool", lambda: server.blocks(limit=5))
    _run_check(
        report,
        "directory",
        "tool",
        lambda: server.directory(limit=5, order="active", local=True),
    )

    # Track which RO tools we explicitly covered
    covered = {r.name.split(":")[0] for r in report.results if r.kind == "tool"}

    for t in tools:
        if not _tool_is_readonly(t):
            continue
        if t.name in covered:
            continue
        fn = fn_by_name.get(t.name)
        if not fn:
            report.add(
                CheckResult(
                    name=t.name,
                    kind="tool",
                    ok=False,
                    detail="registered in MCP but missing Python callable",
                )
            )
            continue
        report.add(
            CheckResult(
                name=t.name,
                kind="tool",
                ok=True,
                detail="no default args probe defined",
                skipped=True,
            )
        )

    # ---- READ_ONLY guard on mutating tools ----
    if read_only:
        sample_writes: List[Tuple[str, Callable[[], Any]]] = [
            ("status_post", lambda: server.status_post(status="MCP live capability probe — should be blocked")),
            ("status_delete", lambda: server.status_delete(status_id=status_id or "1")),
            ("status_favourite", lambda: server.status_favourite(status_id=status_id or "1")),
            ("status_unfavourite", lambda: server.status_unfavourite(status_id=status_id or "1")),
            ("status_reblog", lambda: server.status_reblog(status_id=status_id or "1")),
            ("status_unreblog", lambda: server.status_unreblog(status_id=status_id or "1")),
            ("status_bookmark", lambda: server.status_bookmark(status_id=status_id or "1")),
            ("status_unbookmark", lambda: server.status_unbookmark(status_id=status_id or "1")),
            ("account_follow", lambda: server.account_follow(account_id=other_id or me_id or "1")),
            ("account_unfollow", lambda: server.account_unfollow(account_id=other_id or me_id or "1")),
            ("account_mute", lambda: server.account_mute(account_id=other_id or me_id or "1")),
            ("account_unmute", lambda: server.account_unmute(account_id=other_id or me_id or "1")),
            ("account_block", lambda: server.account_block(account_id=other_id or me_id or "1")),
            ("account_unblock", lambda: server.account_unblock(account_id=other_id or me_id or "1")),
            ("account_update", lambda: server.account_update(display_name="MCP probe")),
            ("notification_dismiss", lambda: server.notification_dismiss(notification_id="1")),
            ("notifications_clear", lambda: server.notifications_clear()),
            ("list_create", lambda: server.list_create(title="MCP probe list")),
            ("list_delete", lambda: server.list_delete(list_id=list_id or "1")),
            ("list_accounts_add", lambda: server.list_accounts_add(list_id=list_id or "1", account_ids=[me_id or "1"])),
            ("list_accounts_delete", lambda: server.list_accounts_delete(list_id=list_id or "1", account_ids=[me_id or "1"])),
            ("poll_vote", lambda: server.poll_vote(poll_id="1", choices=[0])),
            ("follow_request_authorize", lambda: server.follow_request_authorize(account_id=other_id or "1")),
            ("follow_request_reject", lambda: server.follow_request_reject(account_id=other_id or "1")),
            ("media_post", lambda: server.media_post(file_path="/tmp/nonexistent-mcp-probe.png")),
        ]
        for name, fn in sample_writes:
            _assert_write_blocked(report, name, fn)

        # Ensure every RW tool was guard-sampled
        guarded = {r.name.replace("readonly_guard:", "") for r in report.results if r.kind == "guard"}
        for t in tools:
            if _tool_is_readonly(t):
                continue
            report.add(
                CheckResult(
                    name=f"catalog:{t.name}",
                    kind="meta",
                    ok=t.name in guarded,
                    detail="write tool registered and guard-sampled"
                    if t.name in guarded
                    else "write tool registered but NOT guard-sampled",
                )
            )

    _ = tool_by_name
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instance", default=os.getenv("MASTODON_INSTANCE"))
    parser.add_argument("--access-token", default=os.getenv("MASTODON_ACCESS_TOKEN"))
    parser.add_argument("--json-out", help="Write full report JSON here")
    parser.add_argument(
        "--allow-writes",
        action="store_true",
        help="Set READ_ONLY=false (guards skipped; not used for data checks)",
    )
    args = parser.parse_args()

    # Prefer ambient READ_ONLY when not forcing --allow-writes
    env_ro = os.getenv("READ_ONLY", "true").lower() in ("true", "1", "yes")
    read_only = False if args.allow_writes else env_ro

    missing = [
        n
        for n, v in [
            ("--instance/MASTODON_INSTANCE", args.instance),
            ("--access-token/MASTODON_ACCESS_TOKEN", args.access_token),
        ]
        if not v
    ]
    if missing:
        print(f"Missing required config: {', '.join(missing)}", file=sys.stderr)
        return 2

    report = run_scenario(
        args.instance,
        args.access_token,
        read_only=read_only,
    )

    print(f"Mastodon MCP live scenario: {report.instance}")
    print(f"READ_ONLY={str(read_only).lower()}")
    print(f"passed={report.passed} failed={report.failed} skipped={report.skipped}")
    print()
    for r in report.results:
        if r.skipped:
            flag = "SKIP"
        elif r.ok:
            flag = "PASS"
        else:
            flag = "FAIL"
        print(f"  {flag:4} [{r.kind}] {r.name}: {r.detail}")

    if args.json_out:
        out = {
            "instance": report.instance,
            "read_only": read_only,
            "passed": report.passed,
            "failed": report.failed,
            "skipped": report.skipped,
            "results": [r.__dict__ for r in report.results],
        }
        Path(args.json_out).write_text(json.dumps(out, indent=2, ensure_ascii=False, default=str))
        print(f"\nWrote {args.json_out}")

    return 0 if report.failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
