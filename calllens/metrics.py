"""Measurements we can compute directly from the transcript, no AI needed.

These are objective counts. The AI judges things like outcome and sentiment;
the code measures things like who talked more. Keeping the two separate means the
dashboard's numbers don't depend on the AI getting arithmetic right.
"""

from __future__ import annotations

from .schema import Call, CallMetrics, Speaker


def _words(text: str) -> int:
    return len(text.split())


def compute_metrics(call: Call) -> CallMetrics:
    agent = [t for t in call.turns if t.speaker == Speaker.AGENT]
    customer = [t for t in call.turns if t.speaker == Speaker.CUSTOMER]
    tools = [t for t in call.turns if t.speaker == Speaker.TOOL]

    agent_words = sum(_words(t.text) for t in agent)
    customer_words = sum(_words(t.text) for t in customer)
    total = agent_words + customer_words

    stamps = [t.start_s for t in call.turns if t.start_s is not None]
    # Duration = last timestamp + a rough allowance for the last sentence (~2.5 words/sec).
    duration = None
    if stamps:
        duration = max(stamps) + _words(call.turns[-1].text) / 2.5

    return CallMetrics(
        duration_s=round(duration, 1) if duration is not None else None,
        turns=len(agent) + len(customer),
        agent_turns=len(agent),
        customer_turns=len(customer),
        agent_talk_share=round(agent_words / total, 3) if total else 0.0,
        longest_agent_turn_words=max((_words(t.text) for t in agent), default=0),
        customer_questions=sum(t.text.count("?") for t in customer),
        interruptions=sum(1 for t in agent if t.interrupted),
        tool_calls=len(tools),
    )
