"""Turn many CallReports into a table and the dashboard's headline numbers (KPIs)."""

from __future__ import annotations

from collections import Counter

import pandas as pd

from .schema import SUCCESS_OUTCOMES, CallReport, Outcome


def to_frame(reports: list[CallReport]) -> pd.DataFrame:
    """One row per call. This is also what the CSV download contains."""
    rows = []
    for r in reports:
        a, m = r.analysis, r.metrics
        rows.append({
            "call_id": r.call.call_id,
            "outcome": a.outcome.value,
            "sentiment": a.customer_sentiment.value,
            "language": a.language,
            "adherence": a.script_adherence,
            "duration_s": m.duration_s,
            "turns": m.turns,
            "agent_talk_share": m.agent_talk_share,
            "interruptions": m.interruptions,
            "customer_questions": m.customer_questions,
            "objections": ", ".join(a.objections),
            "agent_issues": ", ".join(a.agent_issues),
            "summary": a.summary,
            "next_action": a.next_action,
            "analyzer": r.analyzer,
        })
    return pd.DataFrame(rows)


def kpis(reports: list[CallReport]) -> dict:
    n = len(reports)
    if n == 0:
        return {"calls": 0}
    outcomes = [r.analysis.outcome for r in reports]
    connected = [o for o in outcomes if o != Outcome.NO_CONNECT]
    durations = [r.metrics.duration_s for r in reports if r.metrics.duration_s is not None]
    return {
        "calls": n,
        # Of all calls dialled, how many reached a person?
        "connect_rate": len(connected) / n,
        # Of the calls that connected, how many hit the goal (converted or resolved)?
        "success_rate": (sum(o in SUCCESS_OUTCOMES for o in connected) / len(connected)) if connected else 0.0,
        "avg_duration_s": sum(durations) / len(durations) if durations else None,
        "avg_adherence": sum(r.analysis.script_adherence for r in reports) / n,
        "negative_share": sum(r.analysis.customer_sentiment.value == "NEGATIVE" for r in reports) / n,
        "needs_follow_up": sum(o in {Outcome.FOLLOW_UP, Outcome.ESCALATED, Outcome.UNRESOLVED} for o in outcomes),
    }


def count_labels(reports: list[CallReport], field: str, top: int = 8) -> pd.DataFrame:
    """Most common objections or agent issues across all calls."""
    c = Counter(label.strip().lower() for r in reports for label in getattr(r.analysis, field) if label.strip())
    return pd.DataFrame(c.most_common(top), columns=["label", "calls"])
