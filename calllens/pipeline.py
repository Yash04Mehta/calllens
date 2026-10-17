"""Glue: transcript -> metrics + analysis -> CallReport, with a cache.

The cache stores each AI analysis on disk keyed by a hash of (transcript, business context,
model). Re-running the dashboard on the same calls costs zero API calls, which matters on
Gemini's free tier. Change the business context and the calls get re-analysed.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Callable

from .heuristic import analyze_rules
from .metrics import compute_metrics
from .schema import Call, CallAnalysis, CallReport

Analyzer = Callable[[Call], CallAnalysis]

CACHE_DIR = Path("outputs/cache")


def _key(call: Call, extra: str) -> str:
    return hashlib.sha256((call.as_script() + "\n--\n" + extra).encode("utf-8")).hexdigest()[:16]


def analyze_calls(
    calls: list[Call],
    mode: str = "rules",
    business_context: str = "",
    analyzer: Analyzer | None = None,
    cache_dir: Path | None = CACHE_DIR,
    on_progress: Callable[[int, int], None] | None = None,
) -> tuple[list[CallReport], list[str]]:
    """Analyse every call. Returns (reports, errors). One bad call never stops the batch.

    mode: "rules" (offline) or "gemini". `analyzer` overrides both (used by tests).
    """
    if analyzer is None:
        if mode == "gemini":
            from .llm import analyze_call, model_name

            client = None
            tag = model_name()

            def analyzer(c: Call) -> CallAnalysis:
                nonlocal client
                if client is None:
                    from .llm import _client
                    client = _client()
                return analyze_call(c, business_context, client=client)
        else:
            analyzer, tag = analyze_rules, "rules"
    else:
        tag = getattr(analyzer, "__name__", "custom")

    use_cache = cache_dir is not None and mode == "gemini"
    if use_cache:
        Path(cache_dir).mkdir(parents=True, exist_ok=True)

    reports, errors = [], []
    for i, call in enumerate(calls, 1):
        try:
            analysis = None
            cache_file = Path(cache_dir) / f"{_key(call, business_context + tag)}.json" if use_cache else None
            if cache_file and cache_file.exists():
                analysis = CallAnalysis.model_validate(json.loads(cache_file.read_text(encoding="utf-8")))
            if analysis is None:
                analysis = analyzer(call)
                if cache_file:
                    cache_file.write_text(analysis.model_dump_json(indent=2), encoding="utf-8")
            reports.append(CallReport(call=call, metrics=compute_metrics(call), analysis=analysis,
                                      analyzer="gemini" if mode == "gemini" else "rules"))
        except Exception as exc:
            errors.append(f"{call.call_id}: {exc}")
        if on_progress:
            on_progress(i, len(calls))
    return reports, errors
