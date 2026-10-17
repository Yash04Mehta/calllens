"""Turn transcript files into Call objects.

Supported formats (auto-detected):

1. Aria / voice-agent export:   [00:13] CUSTOMER  Where is my order 101?
                                [00:14] TOOL      get_order_details(101) -> found
2. Simple chat style:           Agent: Hello, this is Riya from SkillPath.
                                [00:05] Customer: Haan, boliye.
3. JSON:                        [{"speaker": "agent", "text": "...", "start": 2.5}, ...]
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from .schema import Call, Speaker, Turn

AGENT_NAMES = {"agent", "aria", "bot", "assistant", "ai", "rep", "sales", "support", "executive"}
CUSTOMER_NAMES = {"customer", "user", "caller", "client", "lead", "prospect", "you"}
TOOL_NAMES = {"tool", "system", "function"}

# [mm:ss] or [hh:mm:ss], optional
TS = r"(?:\[(?P<ts>\d{1,2}:\d{2}(?::\d{2})?)\]\s*)?"
LINE = re.compile(TS + r"(?P<who>[A-Za-z][A-Za-z ._-]{0,30}?)\s*(?::|\s{2,}|\t)\s*(?P<text>.+)$")
INTERRUPTED = re.compile(r"\s*\[interrupted\]\s*$", re.IGNORECASE)
TYPED = re.compile(r"\s*\[typed\]\s*$", re.IGNORECASE)


def _seconds(ts: str | None) -> float | None:
    if not ts:
        return None
    parts = [int(p) for p in ts.split(":")]
    secs = 0
    for p in parts:
        secs = secs * 60 + p
    return float(secs)


def _speaker(name: str) -> Speaker | None:
    n = name.strip().lower()
    first = n.split()[0] if n else n
    if n in AGENT_NAMES or first in AGENT_NAMES:
        return Speaker.AGENT
    if n in CUSTOMER_NAMES or first in CUSTOMER_NAMES:
        return Speaker.CUSTOMER
    if n in TOOL_NAMES or first in TOOL_NAMES:
        return Speaker.TOOL
    return None


def parse_text(text: str, call_id: str, source: str = "") -> Call:
    """Parse a plain-text transcript. Lines that don't start with a known speaker are
    treated as a continuation of the previous turn (long turns often wrap)."""
    turns: list[Turn] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        m = LINE.match(line)
        who = _speaker(m.group("who")) if m else None
        if m and who is not None:
            body = m.group("text").strip()
            interrupted = bool(INTERRUPTED.search(body))
            body = TYPED.sub("", INTERRUPTED.sub("", body)).strip()
            if not body or body == "-":
                continue
            turns.append(Turn(speaker=who, text=body, start_s=_seconds(m.group("ts")), interrupted=interrupted))
        elif turns:
            turns[-1].text += " " + line
    if not turns:
        raise ValueError(f"{source or call_id}: no speaker lines found (expected e.g. 'Agent: ...').")
    return Call(call_id=call_id, source=source, turns=turns)


def parse_json(data: list | dict, call_id: str, source: str = "") -> Call:
    items = data.get("turns", data) if isinstance(data, dict) else data
    turns = []
    for it in items:
        who = _speaker(str(it.get("speaker", "")))
        if who is None or not str(it.get("text", "")).strip():
            continue
        start = it.get("start", it.get("start_s"))
        turns.append(Turn(speaker=who, text=str(it["text"]).strip(),
                          start_s=float(start) if start is not None else None,
                          interrupted=bool(it.get("interrupted", False))))
    if not turns:
        raise ValueError(f"{source or call_id}: JSON has no usable turns.")
    return Call(call_id=call_id, source=source, turns=turns)


def parse_file(path: Path) -> Call:
    path = Path(path)
    content = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        return parse_json(json.loads(content), path.stem, path.name)
    return parse_text(content, path.stem, path.name)


def parse_folder(folder: Path) -> list[Call]:
    files = sorted(p for p in Path(folder).iterdir() if p.suffix.lower() in {".txt", ".json", ".md"})
    return [parse_file(p) for p in files]
