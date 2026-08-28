"""agent/gateway.py — YOUR control plane. CONTRACTS.md section 4, exactly.

READ agent/README.md FIRST — it maps all five files in this directory to what
each is scored on. This file is the one CONTRACTS.md calls "the trusted
envelope's untrusted half": every single MCP / A2A / DISCOVER command your
agent's model wants to make passes through `Gateway.decide` before it is
allowed to happen.

WHY THERE IS NO `execute()` METHOD ON `GatewayContext` (read this before you
go looking for one — there isn't one, and that is not an oversight)
----------------------------------------------------------------------------
CONTRACTS.md section 4's trusted envelope, reproduced here because it is the
one diagram worth memorising:

    [ trusted ]   loop emits a raw action line
         v
    [ trusted ]   INTERCEPT + CANONICALISE -> Command        (kit/loop/agent.py)
         v
    [ UNTRUSTED ] Gateway.decide(cmd) -> Decision             <- THIS FILE
         v
    [ trusted ]   ENFORCE: honour the Decision, meter it,
                  apply the active mutation, execute the
                  ToolCall or refuse it                       (the arena)
         v
    [ trusted ]   RECORD the authoritative L1 event, then
                  RENDER the Observation                      (the arena)
         v
    [ trusted ]   the model sees the Observation

`decide()` returns a *decision*, never a *result*. You cannot reach a tool
server, a file, a socket, or a clock from in here — there is nothing to
call. Two things follow from that, and both matter more than they look:

  1. YOUR TRACE CANNOT BE FORGED. Every `command` / `decision` / `enforced`
     / `tool_call` / `tool_result` L1 event (CONTRACTS.md 5.2) is written by
     the arena, from what the arena itself actually did — never from
     anything you claimed happened. A student gateway that wanted to lie
     about having blocked an attack ("I totally denied that, trust me")
     simply has no channel to lie through: the only thing you ever hand
     back is this one small `Decision` value, and the arena is the one that
     turns it into history.
  2. NOBODY CAN ACCUSE YOU OF A CALL YOU DID NOT AUTHORISE, either. Because
     `decide()` is the ONLY door a command can walk through on its way to
     actually running, a prosecutor's `enforcement_failure` claim against
     you has exactly one thing to point at: the `Decision` you returned for
     that specific `cmd_id`. There is no ambiguity about "maybe the loop
     called the tool directly" — CONTRACTS.md 4.2 removed that path on
     purpose, and kit/loop/agent.py's own module docstring names the same
     invariant from the other side (the loop never imports this module,
     never sees a `Decision`, never executes anything itself).

The cost of that guarantee is that this file is PURE: synchronous, no I/O,
no threads, no `sleep`, 250 ms wall-clock deadline (RULES.md section 3).
Raising anything, returning something that is not a valid `Decision`, or
missing the deadline is treated by the arena as a DENIED command PLUS a 2
credit penalty PLUS an `integrity` event that hands the prosecutor a free
`enforcement_failure` — CONTRACTS.md 4.1's charging table, reproduced in
agent/README.md's own table. Getting this file to just plainly return valid
`Decision` values, every time, is worth more than getting it clever.

THE IMPLEMENTED SHAPE
---------------------
`decide()` is structured as ROUTE, ADMIT, AUTHORIZE, and BUDGET. It rewrites
deprecated or excessive-mask calls, validates leases, scopes, learner
ownership, A2A peer admission, write preconditions and observed provenance,
then enforces a conservative per-round credit ceiling. Every negative branch
returns a valid, zero-credit denial; it never raises on an untrusted command.

ONE THING WORTH INTERNALISING BEFORE YOU WRITE YOUR FIRST REAL CHECK:
`verdict="deny"` costs the CALLER (your own team) **zero credits** —
CONTRACTS.md 4.1's charging table has exactly one $0 row, and it is this
one. Refusing to make a call you cannot justify is FREE. That makes
abstention a real strategy, not a luxury you can't afford: a `deny` you can
defend beats a `forward` you can't, every time a prosecutor is watching.

Stdlib only. No network, no randomness, no wall-clock reads, no sleeping —
none of that would even survive the kernel sandbox (CONTRACTS.md 12), but
the point is this file has no reason to want any of it in the first place.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Protocol, runtime_checkable

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from agent.strategy import (
    BudgetPacer,
    is_catalog_trap,
    successor_of,
)

# kit.mcp.types is a collaborator's file (workspace hard rule 2: import it,
# degrade gracefully). It is present as of this writing and is core, stable
# infrastructure (CONTRACTS.md 3.1) — but this module must still not fail to
# IMPORT if a concurrent edit ever breaks it transiently. When it is
# unavailable, `Decision.call` type-checking is skipped (not enforced), and
# `Gateway.decide` falls back to a minimal local dict-shaped stand-in so the
# rest of this file — everything that does not need a *real* ToolCall — still
# runs.
try:
    from kit.mcp.types import ToolCall
    _TOOLCALL_AVAILABLE = True
except ImportError:  # pragma: no cover - collaborator file
    ToolCall = Any  # type: ignore[assignment, misc]
    _TOOLCALL_AVAILABLE = False

try:
    from kit.mcp.specs import cost as _tool_cost
except ImportError:  # pragma: no cover - collaborator file
    _tool_cost = None

# kit.loop.agent is also a collaborator's file, used only by this module's
# own __main__ demo (to build real Commands the same way the arena's trusted
# canonicaliser would) — never by decide() itself, which never touches the
# loop. Degraded the same way.
try:
    from kit.loop.agent import canonicalise_action as _canonicalise_action
except ImportError:  # pragma: no cover - collaborator file
    _canonicalise_action = None

from agent.telemetry import RecordingGatewayContext, Telemetry

__all__ = [
    "COMMAND_KINDS",
    "DECISION_VERDICTS",
    "Command",
    "Decision",
    "GatewayContext",
    "Gateway",
]

# CONTRACTS.md 4.1: `Command.kind` — "mcp" | "a2a" | "discover". An "answer"
# action is NEVER a Command (kit/loop/agent.py's own module docstring: "an
# answer is not a tool call routed to a server, so it never becomes a
# Command at all") — it is handled entirely by the loop/arena and never
# reaches `Gateway.decide`.
COMMAND_KINDS: frozenset[str] = frozenset({"mcp", "a2a", "discover"})

# CONTRACTS.md 4.1: `Decision.verdict` — the closed three-member set.
DECISION_VERDICTS: frozenset[str] = frozenset({"forward", "deny", "rewrite"})


@dataclass(frozen=True, slots=True)
class Command:
    """CONTRACTS.md 4.1, field for field — "canonicalised by the arena
    BEFORE the student sees it". You never build one of these from your own
    agent's raw text; the arena's canonicaliser (kit/loop/agent.py's
    `canonicalise_action`, run inside the trusted envelope) already did that
    work and minted `cmd_id` by the time `decide()` sees it. The
    `from_action_dict` classmethod below exists only so this file's own demo
    (and your local tests, if you write any) can build a realistic `Command`
    without duplicating the arena's canonicalisation logic."""

    cmd_id: str
    kind: str  # "mcp" | "a2a" | "discover" — see COMMAND_KINDS
    raw: str
    server: str
    tool: str
    args: dict
    fields: tuple[str, ...]
    headers: dict
    lease_id: str | None
    call_index: int

    def __post_init__(self) -> None:
        if not isinstance(self.cmd_id, str) or not self.cmd_id:
            raise ValueError(f"Command.cmd_id must be a non-empty str, got {self.cmd_id!r}")
        if self.kind not in COMMAND_KINDS:
            raise ValueError(f"Command.kind must be one of {sorted(COMMAND_KINDS)}, got {self.kind!r}")
        if not isinstance(self.server, str) or not self.server:
            raise ValueError(f"Command.server must be a non-empty str, got {self.server!r}")
        if not isinstance(self.tool, str) or not self.tool:
            raise ValueError(f"Command.tool must be a non-empty str, got {self.tool!r}")
        if not isinstance(self.args, dict):
            raise ValueError(f"Command.args must be a dict, got {type(self.args).__name__}")
        if not isinstance(self.headers, dict):
            raise ValueError(f"Command.headers must be a dict, got {type(self.headers).__name__}")
        if (
            not isinstance(self.call_index, int)
            or isinstance(self.call_index, bool)
            or self.call_index < 0
        ):
            raise ValueError(f"Command.call_index must be a non-negative int, got {self.call_index!r}")

    @classmethod
    def from_action_dict(cls, action: Mapping[str, Any], *, cmd_id: str) -> "Command":
        """Build a `Command` from the dict shape `kit.loop.agent.canonicalise_action`
        returns (`kind, raw, server, tool, args, fields, headers, lease_id,
        call_index` — everything except the arena-minted `cmd_id`, supplied
        here as a keyword). Raises `ValueError` if `action["kind"] ==
        "answer"` — an answer is never a Command (see the module docstring).
        This is a convenience for tests/demos, not something the real arena
        calls: the trusted envelope mints `cmd_id` itself and constructs the
        real `Command` on its own side of the boundary."""
        kind = action.get("kind")
        if kind == "answer":
            raise ValueError(
                "an 'answer' action never becomes a Command (kit/loop/agent.py: "
                "\"an answer is not a tool call routed to a server\") — do not "
                "route it through Gateway.decide at all"
            )
        return cls(
            cmd_id=cmd_id,
            kind=kind,
            raw=action["raw"],
            server=action["server"],
            tool=action["tool"],
            args=dict(action.get("args", {})),
            fields=tuple(action.get("fields", ())),
            headers=dict(action.get("headers", {})),
            lease_id=action.get("lease_id"),
            call_index=action.get("call_index", 0),
        )

    def to_dict(self) -> dict:
        return {
            "cmd_id": self.cmd_id,
            "kind": self.kind,
            "raw": self.raw,
            "server": self.server,
            "tool": self.tool,
            "args": dict(self.args),
            "fields": list(self.fields),
            "headers": dict(self.headers),
            "lease_id": self.lease_id,
            "call_index": self.call_index,
        }


@dataclass(frozen=True, slots=True)
class Decision:
    """CONTRACTS.md 4.1, field for field.

    Validated strictly (`__post_init__`) because a *structurally* invalid
    `Decision` is charged exactly like a raised exception — CONTRACTS.md
    4.1's charging table: "malformed Decision (schema-invalid) -> 2 cr
    penalty, command denied." Failing loudly HERE, in your own process
    during development, is strictly better than discovering it live in a
    duel as an unexplained penalty.

    `verdict == "deny"` requires a non-empty `reason` (CONTRACTS.md 4.1:
    "required when verdict == 'deny'; shown in the combat log") and
    forbids `call` — a real denial has nothing left to carry out.
    `verdict` in `("forward", "rewrite")` requires `call` to be set — the
    arena executes exactly that `ToolCall`, nothing else, per the trusted
    envelope's whole point (see the module docstring)."""

    verdict: str  # "forward" | "deny" | "rewrite" — see DECISION_VERDICTS
    reason: str | None = None
    call: "ToolCall | None" = None
    quarantine: bool = False
    note: str | None = None

    def __post_init__(self) -> None:
        if self.verdict not in DECISION_VERDICTS:
            raise ValueError(
                f"Decision.verdict must be one of {sorted(DECISION_VERDICTS)}, got {self.verdict!r}"
            )
        if self.verdict == "deny":
            if not isinstance(self.reason, str) or not self.reason.strip():
                raise ValueError("Decision.verdict=='deny' requires a non-empty 'reason'")
            if self.call is not None:
                raise ValueError("Decision.verdict=='deny' must not carry a 'call' — there is nothing to run")
        else:  # forward | rewrite
            if self.call is None:
                raise ValueError(f"Decision.verdict=={self.verdict!r} requires 'call' to be set")
            if _TOOLCALL_AVAILABLE and not isinstance(self.call, ToolCall):
                raise ValueError(
                    f"Decision.call must be a kit.mcp.types.ToolCall instance, got {type(self.call).__name__}"
                )
        if not isinstance(self.quarantine, bool):
            raise ValueError(f"Decision.quarantine must be a bool, got {self.quarantine!r}")
        if self.note is not None and not isinstance(self.note, str):
            raise ValueError(f"Decision.note must be a str or None, got {self.note!r}")

    def to_dict(self) -> dict:
        return {
            "verdict": self.verdict,
            "reason": self.reason,
            "call": self.call.to_dict() if self.call is not None and hasattr(self.call, "to_dict") else self.call,
            "quarantine": self.quarantine,
            "note": self.note,
        }


@runtime_checkable
class GatewayContext(Protocol):
    """CONTRACTS.md 4.2 — "read-only, arena-provided". Note what this is
    NOT: unlike `Command`/`Decision` above, CONTRACTS.md writes this as a
    plain `class`, not a `@dataclass(frozen=True, ...)`. That is a real
    signal, not an inconsistency — a `GatewayContext` is a LIVE VIEW the
    arena owns and updates as the duel actually progresses (credits get
    spent, rounds advance, leases mint and expire), not a frozen snapshot
    you receive once. This file therefore models it as a structural
    `Protocol`: the arena hands your `Gateway.__init__` some object shaped
    like this (see `agent/telemetry.py`'s `RecordingGatewayContext` for a
    concrete example your own tests/demos can build), and you read
    `self.ctx.<field>` fresh every time `decide()` runs — NEVER cache a
    context field into a local variable across calls; if you need memory
    that persists across the duel (a cache, a budget tracker, drift notes),
    keep it on YOUR OWN `Gateway` instance attributes instead (CONTRACTS.md
    4.3: "Gateway is instantiated once per duel... persists across the 10
    rounds").

    There is deliberately no `execute` method here — see the module
    docstring's "WHY THERE IS NO execute() METHOD" section. `emit` is the
    ONE thing you can do that has any side effect at all, and it writes to
    `own_telemetry` only: your own side, never scored, never shown to the
    opponent (`agent/telemetry.py` wraps it)."""

    act: str  # "learner:sv-0417" — WHOM YOU SERVE. Authority derives from this.
    sub: str  # "agent:vlearn-tutor" — what you are. NOT authority.
    scopes: frozenset[str]  # {"wiki.read"}; writes need "wiki.write:progress" etc.
    credits: int  # remaining this duel
    round: int
    call_index: int
    leases: tuple[str, ...]  # live lease ids, arena-tracked
    history: tuple[Mapping[str, Any], ...]  # YOUR OWN prior (Command, Decision, outcome) triples this duel

    def emit(self, name: str, **payload: Any) -> None: ...


def _normalize_learner(val: str | None) -> str | None:
    if not val:
        return None
    s = val.strip().lower()
    if s.startswith("learner:"):
        return s[len("learner:"):]
    return s


WRITE_TOOLS = {
    ("progress", "record_mastery"),
    ("content", "flag_stale_slide"),
    ("content", "file_content_bug"),
}
A2A_SERVERS = {"curriculum-analyst", "citation-checker", "roster"}


class Gateway:
    """The control plane. One instance per duel (CONTRACTS.md 4.3) — built
    once at duel start with a `GatewayContext`, then asked to `decide()` on
    every MCP/A2A/DISCOVER command either side of the duel makes for all 10
    rounds. See the module docstring for the trusted-envelope diagram and
    why there is no `execute()` to call instead.
    """

    def __init__(self, ctx: GatewayContext) -> None:
        self.ctx = ctx
        self._telemetry = Telemetry(ctx)
        self._pacer = BudgetPacer()
        self._etags: dict[str, str] = {}
        self._admitted_cards: dict[str, dict] = {}
        self._credits_authorised: int = 0
        self._denied_cmd_ids: set[str] = set()
        self._seen_idempotency_keys: set[str] = set()
        self._drift_notes: dict[str, bool] = {}
        self._preferred_replicas: dict[str, str] = {}
        self._spent_by_round: dict[int, int] = {}
        self._initial_credits: int = max(0, int(getattr(ctx, "credits", 0)))

    def note_provenance(self, anchor: str, etag: str) -> None:
        """Feed fresh etag from provenance check before writes."""
        self._etags[anchor] = etag

    def note_card(self, server: str, card: dict) -> None:
        """Feed verified peer Agent Card from registry."""
        self._admitted_cards[server] = card

    def note_drift(self, path_id: str, drifting: bool, fresher_replica: str | None = None) -> None:
        """Feed a drift observation and, when known, its fresher replica.

        Merely knowing that two replicas differ is not enough to assume that
        ``w`` is newer than ``c``.  Routing changes only after the caller has
        supplied an explicit, validated winner.
        """
        self._drift_notes[path_id] = drifting
        if not drifting:
            self._preferred_replicas.pop(path_id, None)
        elif fresher_replica in {"w", "c"}:
            self._preferred_replicas[path_id] = fresher_replica

    def decide(self, cmd: Command) -> Decision:
        """SYNCHRONOUS. PURE. NO I/O. 250 ms wall (RULES.md section 3)."""
        self._telemetry.decision_seen(cmd)

        # ------------------------------------------------------------------
        # JOB 1 — ROUTE: is this the right SERVER/REPLICA for this command?
        # Refuse routes smuggled through the request body (header_spoof defense).
        body_route = cmd.args.get("route") or cmd.args.get("_route") or cmd.args.get("replica")
        if body_route:
            return self.deny(cmd, reason="header_spoof: route declared in body instead of header")

        routed = cmd
        succ = successor_of(cmd.server, cmd.tool)
        if succ is not None:
            succ_server, succ_tool = succ
            fields = cmd.fields
            if (cmd.server, cmd.tool) == ("slides", "search") and (succ_server, succ_tool) == ("slides", "query"):
                fields = tuple(f for f in cmd.fields if f in ("title", "body", "score")) or ("title",)
            routed = replace(routed, server=succ_server, tool=succ_tool, fields=fields)

        path_id = cmd.args.get("path_id")
        preferred_replica = self._preferred_replicas.get(str(path_id)) if path_id else None
        if preferred_replica:
            if routed.headers.get("mcp-replica") != preferred_replica:
                new_headers = dict(routed.headers)
                new_headers["mcp-replica"] = preferred_replica
                routed = replace(routed, headers=new_headers)

        # ------------------------------------------------------------------
        # JOB 2 — ADMIT: is this call worth letting through AT ALL?
        # 1. Lease check on slides.get_frame
        if routed.server == "slides" and routed.tool == "get_frame":
            if not routed.lease_id:
                return self.deny(cmd, reason="protocol_misuse: slides.get_frame requires a valid lease_id")
            if not self.ctx.leases or routed.lease_id not in self.ctx.leases:
                return self.deny(
                    cmd,
                    reason=f"protocol_misuse: lease_id {routed.lease_id!r} is not active in context leases {self.ctx.leases!r}"
                )

        # 2. Write headers check (If-Match & Idempotency-Key)
        is_write_cmd = (
            (routed.server, routed.tool) in WRITE_TOOLS
            or bool(routed.headers.get("if-match"))
            or bool(routed.headers.get("idempotency-key"))
        )
        if is_write_cmd:
            if not routed.headers.get("if-match"):
                return self.deny(cmd, reason="write_violation: write commands require If-Match header")
            if not routed.headers.get("idempotency-key"):
                return self.deny(cmd, reason="write_violation: write commands require Idempotency-Key header")
            idem_key = str(routed.headers["idempotency-key"])
            if idem_key in self._seen_idempotency_keys:
                return self.deny(cmd, reason=f"write_violation: duplicate Idempotency-Key {idem_key!r} already used this duel")

        # ------------------------------------------------------------------
        # JOB 3 — AUTHORIZE: authority & scope
        # 1. Scope check for writes
        if is_write_cmd:
            required_scope = f"wiki.write:{routed.server}"
            scopes = frozenset(self.ctx.scopes or ())
            if required_scope not in scopes:
                return self.deny(
                    cmd,
                    reason=f"authority_exceeded: write requires scope {required_scope!r} not in ctx.scopes {sorted(scopes)!r}"
                )

        # 2. Learner identity check (act ownership)
        for key in ("learner", "learner_id", "target", "subject", "act"):
            target_learner = routed.args.get(key)
            if target_learner and self.ctx.act:
                if _normalize_learner(str(target_learner)) != _normalize_learner(self.ctx.act):
                    return self.deny(
                        cmd,
                        reason=f"authority_exceeded: target {target_learner!r} != caller act {self.ctx.act!r}"
                    )

        # 3. A2A Peer & Delegation verification
        if routed.server in A2A_SERVERS or routed.kind == "a2a":
            if routed.server not in A2A_SERVERS:
                return self.deny(cmd, reason=f"forged_card: unknown A2A peer {routed.server!r}")

            # Audience check
            aud = routed.headers.get("aud")
            if not aud:
                return self.deny(cmd, reason="identity: A2A delegation requires an aud header")
            if aud not in (routed.server, f"mcp:{routed.server}", f"a2a:{routed.server}"):
                return self.deny(
                    cmd,
                    reason=f"identity: delegation aud {aud!r} does not match target server {routed.server!r}"
                )

            # Card & Skill check
            card = self._admitted_cards.get(routed.server)
            if card is None:
                return self.deny(cmd, reason=f"forged_card: no verified peer card admitted for {routed.server!r}")
            if card.get("verified") is not True:
                return self.deny(cmd, reason=f"forged_card: peer card for {routed.server!r} not verified")
            raw_skills = card.get("skills")
            if not isinstance(raw_skills, (list, tuple, set, frozenset)):
                return self.deny(cmd, reason=f"forged_card: peer card for {routed.server!r} has no valid skills list")
            declared_skills = {str(skill) for skill in raw_skills}
            if routed.tool not in declared_skills:
                return self.deny(cmd, reason=f"forged_card: skill {routed.tool!r} not declared on peer card")

        # A write must be based on a provenance value observed by this
        # gateway, not merely on an attacker-supplied If-Match string.
        if is_write_cmd:
            target_anchor = next(
                (str(routed.args[key]) for key in ("anchor", "kc", "concept") if routed.args.get(key)),
                None,
            )
            expected_etag = self._etags.get(target_anchor or "")
            supplied_etag = str(routed.headers.get("if-match", ""))
            if expected_etag is None:
                return self.deny(cmd, reason="write_violation: no fresh provenance etag recorded for write target")
            if supplied_etag != expected_etag:
                return self.deny(
                    cmd,
                    reason=f"write_violation: If-Match does not match fresh provenance for {target_anchor!r}",
                )

        # ------------------------------------------------------------------
        # JOB 4 — BUDGET: field mask and pacing
        if is_catalog_trap(routed.server, routed.tool, routed.fields):
            if (routed.server, routed.tool) == ("registry", "list_servers"):
                routed = replace(routed, fields=("name",))
            elif (routed.server, routed.tool) == ("glossary", "list_terms"):
                routed = replace(routed, fields=("term",))
        elif routed.fields == ("*",):
            if (routed.server, routed.tool) == ("slides", "get_frame"):
                routed = replace(routed, fields=("body", "title"))
            elif (routed.server, routed.tool) == ("slides", "query"):
                routed = replace(routed, fields=("body", "title"))
            elif (routed.server, routed.tool) == ("registry", "list_servers"):
                routed = replace(routed, fields=("name",))
            elif (routed.server, routed.tool) == ("glossary", "list_terms"):
                routed = replace(routed, fields=("term",))

        # Price the least expensive possible result (one row). The trusted
        # executor will charge the exact row count; this pre-flight ceiling
        # prevents obviously unsustainable plans without inventing results.
        current_round = getattr(self.ctx, "round", 1) or 1
        try:
            estimated_cost = (
                _tool_cost(routed.server, routed.tool, tuple(routed.fields), n_rows=1)
                if _tool_cost is not None
                else 1
            )
        except (KeyError, TypeError, ValueError):
            return self.deny(cmd, reason=f"protocol_misuse: unknown tool or invalid field mask {routed.server}.{routed.tool}")

        spent_this_round = self._spent_by_round.get(current_round, 0)
        if spent_this_round + estimated_cost > 11:
            return self.deny(cmd, reason="wasteful: estimated round spend would exceed 11 credits")
        available_credits = min(
            max(0, int(getattr(self.ctx, "credits", 0))),
            max(0, self._initial_credits - self._credits_authorised),
        )
        if estimated_cost > available_credits:
            return self.deny(cmd, reason="wasteful: estimated call cost exceeds remaining authorised credits")

        self._spent_by_round[current_round] = spent_this_round + estimated_cost
        self._credits_authorised += estimated_cost
        self._pacer.record_spend(current_round, estimated_cost)
        if is_write_cmd:
            self._seen_idempotency_keys.add(str(routed.headers["idempotency-key"]))

        verdict = "rewrite" if routed != cmd else "forward"
        call = self._to_tool_call(routed)
        decision = Decision(verdict=verdict, call=call)
        self._telemetry.decision_made(cmd, decision)
        return decision

    def deny(self, cmd: Command, reason: str) -> Decision:
        """Build and record a structurally valid, zero-credit denial."""
        self._denied_cmd_ids.add(cmd.cmd_id)
        decision = Decision(verdict="deny", reason=reason)
        self._telemetry.decision_made(cmd, decision)
        return decision

    def _to_tool_call(self, cmd: Command) -> "ToolCall":
        """`Command` -> the `ToolCall` (CONTRACTS.md 3.1) the arena will
        actually execute on a `forward`/`rewrite` verdict. When
        `kit.mcp.types` is unavailable (see the module-level import guard),
        falls back to a plain dict carrying the identical fields — `Decision`
        accepts it either way (the `ToolCall` isinstance check inside
        `Decision.__post_init__` only runs when the real class loaded)."""
        fields = {
            "server": cmd.server,
            "tool": cmd.tool,
            "args": dict(cmd.args),
            "fields": cmd.fields,
            "headers": dict(cmd.headers),
            "lease_id": cmd.lease_id,
            "call_index": cmd.call_index,
        }
        if _TOOLCALL_AVAILABLE:
            return ToolCall(**fields)
        return fields  # type: ignore[return-value]


if __name__ == "__main__":
    print("=== agent.gateway: Command / Decision validation ===\n")

    good_cmd = Command(
        cmd_id="cmd:0000",
        kind="mcp",
        raw="MCP slides.get_frame anchor=Frame:3f2a9c11/w/041 fields=title,body lease=lse_7f21",
        server="slides",
        tool="get_frame",
        args={"anchor": "Frame:3f2a9c11/w/041"},
        fields=("body", "title"),
        headers={},
        lease_id="lse_7f21",
        call_index=0,
    )
    print(f"  Command constructed: {good_cmd}")
    assert good_cmd.kind == "mcp"

    print("\n  Rejection demo (each must raise ValueError):")

    def _expect_value_error(label: str, fn) -> None:
        try:
            fn()
        except ValueError as exc:
            print(f"    [{label:38}] -> ValueError: {exc}")
        else:
            raise AssertionError(f"expected ValueError for case {label!r}")

    _expect_value_error("Command.kind == 'answer'", lambda: Command(
        cmd_id="cmd:0001", kind="answer", raw="x", server="slides", tool="get_frame",
        args={}, fields=(), headers={}, lease_id=None, call_index=0,
    ))
    _expect_value_error("Decision verdict='deny' with no reason", lambda: Decision(verdict="deny"))
    _expect_value_error(
        "Decision verdict='forward' with no call", lambda: Decision(verdict="forward")
    )
    _expect_value_error(
        "Decision verdict='deny' carrying a call",
        lambda: Decision(verdict="deny", reason="nope", call={"server": "x", "tool": "y"}),
    )
    _expect_value_error("Decision verdict='?' unknown", lambda: Decision(verdict="???"))

    print("\n=== Command.from_action_dict — real canonicaliser integration ===\n")
    if _canonicalise_action is None:
        print("  kit.loop.agent not importable yet — skipping the live canonicaliser demo")
        demo_commands: list[Command] = [good_cmd]
    else:
        raw_actions = [
            "MCP registry.provenance anchor=Frame:3f2a9c11/w/041 fields=etag",
            'MCP slides.query q="streamable http replaces http+sse" fields=title,body',
            "A2A curriculum-analyst.which_days_cover concept=Concept:streamable-http fields=anchor,course_day,track",
            "DISCOVER registry.list_servers fields=name",
        ]
        demo_commands = []
        for i, raw in enumerate(raw_actions):
            action = _canonicalise_action(raw, call_index=i)
            cmd = Command.from_action_dict(action, cmd_id=f"cmd:{i:04d}")
            print(f"  {raw!r}\n    -> {cmd.kind}: {cmd.server}.{cmd.tool} fields={cmd.fields}")
            demo_commands.append(cmd)
        assert {c.kind for c in demo_commands} == {"mcp", "a2a", "discover"}

        answer_action = _canonicalise_action(
            'ANSWER {"text": "day 26, track P2T2"}', call_index=None
        )
        try:
            Command.from_action_dict(answer_action, cmd_id="cmd:9999")
        except ValueError as exc:
            print(f"\n  an 'answer' action correctly refuses to become a Command: {exc}")
        else:
            raise AssertionError("expected ValueError for an 'answer' action")

    print("\n=== Gateway.decide — valid calls pass the defensive gateway ===\n")
    ctx = RecordingGatewayContext(
        act="learner:sv-0401",
        sub="agent:demo-team",
        scopes=frozenset({"wiki.read"}),
        credits=100,
        round=1,
        call_index=0,
        leases=(),
        history=(),
    )
    assert isinstance(ctx, GatewayContext), "RecordingGatewayContext must structurally satisfy GatewayContext"
    gw = Gateway(ctx)
    gw.note_card(
        "curriculum-analyst",
        {"verified": True, "skills": ["which_days_cover"]},
    )
    for i, cmd in enumerate(demo_commands, start=1):
        ctx.round = i
        if cmd.kind == "a2a":
            cmd = replace(cmd, headers={"aud": f"a2a:{cmd.server}"})
        decision = gw.decide(cmd)
        print(f"  decide({cmd.server}.{cmd.tool}) -> verdict={decision.verdict!r} quarantine={decision.quarantine}")
        assert decision.verdict == "forward"
        assert decision.call is not None
        call_dict = decision.call.to_dict() if hasattr(decision.call, "to_dict") else decision.call
        assert call_dict["server"] == cmd.server
        assert call_dict["tool"] == cmd.tool
        assert tuple(call_dict["fields"]) == cmd.fields

    print(f"\n=== Gateway.deny — the free-abstention path ===\n")
    denial = gw.deny(demo_commands[0], reason="demo: withholding pending a fresher registry.provenance read")
    print(f"  gw.deny(...) -> verdict={denial.verdict!r} reason={denial.reason!r} call={denial.call!r}")
    assert denial.verdict == "deny"
    assert denial.call is None
    assert demo_commands[0].cmd_id in gw._denied_cmd_ids

    print(f"\n=== own_telemetry — recorded on YOUR side only, never shown to the opponent ===\n")
    print(f"  {len(ctx.events)} events recorded on this ctx this run:")
    for ev in ctx.events:
        print(f"    {ev['name']}: {sorted(ev['payload'].keys())}")
    assert len(ctx.events) >= len(demo_commands) * 2 + 1  # decision_seen + decision_made per call, plus the deny

    print("\nAll agent/gateway.py demos passed.")
