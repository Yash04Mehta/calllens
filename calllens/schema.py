"""Data shapes used everywhere in CallLens.

A Call is a list of Turns (who spoke, what they said, when).
A CallAnalysis is what the AI (or the offline rules) concludes about the call.
A CallReport bundles the call, its measured metrics and its analysis.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class Speaker(str, Enum):
    AGENT = "agent"
    CUSTOMER = "customer"
    TOOL = "tool"          # system events such as an order lookup in an AI agent's transcript


class Turn(BaseModel):
    speaker: Speaker
    text: str
    start_s: float | None = None     # seconds from call start, if the transcript has timestamps
    interrupted: bool = False        # agent was cut off mid-sentence (barge-in)


class Call(BaseModel):
    call_id: str
    source: str = ""                 # file name it came from
    turns: list[Turn]

    def text_for(self, speaker: Speaker) -> str:
        return " ".join(t.text for t in self.turns if t.speaker == speaker)

    def as_script(self) -> str:
        """Readable transcript, used as the input to the AI analyzer."""
        lines = []
        for t in self.turns:
            ts = f"[{int(t.start_s // 60):02d}:{int(t.start_s % 60):02d}] " if t.start_s is not None else ""
            cut = " [interrupted]" if t.interrupted else ""
            lines.append(f"{ts}{t.speaker.value.upper()}: {t.text}{cut}")
        return "\n".join(lines)


class Outcome(str, Enum):
    CONVERTED = "CONVERTED"                  # goal achieved: demo booked, sale, appointment
    RESOLVED = "RESOLVED"                    # support call: the customer's issue was handled
    FOLLOW_UP = "FOLLOW_UP"                  # interested, callback or follow-up agreed
    NOT_INTERESTED = "NOT_INTERESTED"        # declined
    ESCALATED = "ESCALATED"                  # handed to a human / ticket raised
    UNRESOLVED = "UNRESOLVED"                # call ended without resolving anything
    NO_CONNECT = "NO_CONNECT"                # no answer, voicemail, wrong number


SUCCESS_OUTCOMES = {Outcome.CONVERTED, Outcome.RESOLVED}


class Sentiment(str, Enum):
    POSITIVE = "POSITIVE"
    NEUTRAL = "NEUTRAL"
    NEGATIVE = "NEGATIVE"


class CallAnalysis(BaseModel):
    """What we want to know about every call. Also used as the AI's output schema."""

    outcome: Outcome = Field(description="Final result of the call.")
    customer_sentiment: Sentiment = Field(description="Customer's overall mood by the end of the call.")
    objections: list[str] = Field(default_factory=list, description=(
        "Customer objections or concerns, each as a short lowercase label such as 'price', "
        "'timing', 'trust', 'already using competitor', 'needs to ask family'."))
    agent_issues: list[str] = Field(default_factory=list, description=(
        "Mistakes by the agent, each a short label such as 'talked over customer', 'gave wrong information', "
        "'missed qualifying question', 'ended call abruptly', 'repeated itself', 'off-script'."))
    script_adherence: int = Field(ge=1, le=5, description="1 = ignored the call goals, 5 = followed them fully.")
    language: str = Field(description="Main language of the call: English, Hindi, Hinglish or other.")
    summary: str = Field(description="Two sentences max, past tense, factual.")
    next_action: str = Field(description="One concrete next step for the team, or 'none'.")


class CallMetrics(BaseModel):
    duration_s: float | None
    turns: int
    agent_turns: int
    customer_turns: int
    agent_talk_share: float          # share of words spoken by the agent (0..1)
    longest_agent_turn_words: int
    customer_questions: int
    interruptions: int
    tool_calls: int


class CallReport(BaseModel):
    call: Call
    metrics: CallMetrics
    analysis: CallAnalysis
    analyzer: str                    # "gemini" or "rules"
