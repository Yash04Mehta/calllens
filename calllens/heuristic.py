"""Offline rule-based analyzer: keyword matching, no AI and no API key.

It is deliberately simple. It exists so that:
  * the dashboard works with zero setup (demo mode, tests, no internet), and
  * there is a baseline to compare the AI against ("how much better is Gemini than keywords?").
It understands English plus common Hinglish words. Its output is marked analyzer="rules"
everywhere in the UI so nobody mistakes it for the AI's judgement.
"""

from __future__ import annotations

import re

from .metrics import compute_metrics
from .schema import Call, CallAnalysis, Outcome, Sentiment, Speaker

# Ordered: the first matching rule decides the outcome.
OUTCOME_RULES: list[tuple[Outcome, list[str]]] = [
    (Outcome.NO_CONNECT, ["voicemail", "wrong number", "not reachable", "galat number", "leave a message"]),
    (Outcome.ESCALATED, ["escalate", "senior", "supervisor", "manager se baat", "raise a ticket", "ticket raised",
                         "human agent", "transfer you", "connect you to"]),
    (Outcome.CONVERTED, ["booked", "confirmed your", "slot is confirmed", "see you on", "payment received",
                         "demo is scheduled", "registered", "book kar", "enrolled", "pakka"]),
    (Outcome.RESOLVED, ["has been cancelled", "is cancelled", "refund has been", "address updated",
                        "has been updated", "anything else i can help", "issue is resolved", "ho gaya"]),
    (Outcome.FOLLOW_UP, ["call you back", "call back", "callback", "send you the details", "send details",
                         "whatsapp", "think about it", "baad mein", "kal call", "follow up", "follow-up"]),
    (Outcome.NOT_INTERESTED, ["not interested", "no thanks", "don't call", "do not call", "nahi chahiye",
                              "interest nahi", "remove my number"]),
]

OBJECTION_RULES: dict[str, list[str]] = {
    "price": ["expensive", "costly", "price", "fees", "mehenga", "budget", "discount", "kitna"],
    "timing": ["busy", "no time", "later", "next month", "abhi nahi", "time nahi", "exams"],
    "trust": ["scam", "fraud", "genuine", "reviews", "how do i know", "bharosa", "fake"],
    "already using competitor": ["already enrolled", "already using", "another course", "dusra", "competitor"],
    "needs to ask family": ["ask my parents", "ask my husband", "ask my wife", "family se", "papa se", "discuss at home"],
    "delivery delay": ["late", "delayed", "not delivered", "where is my order", "abhi tak nahi"],
}

POSITIVE = ["thank you", "thanks", "great", "perfect", "awesome", "helpful", "sounds good", "badhiya",
            "accha", "dhanyavaad", "shukriya", "sounds fair"]
NEGATIVE = ["angry", "frustrated", "worst", "terrible", "useless", "ridiculous", "irritating", "pathetic",
            "bakwas", "fed up", "not happy", "disappointed", "waste", "pressure", "never mind"]

HINDI_WORDS = {"haan", "nahi", "kya", "hai", "hain", "mera", "meri", "mujhe", "aap", "kar", "karo", "ji",
               "theek", "accha", "abhi", "kal", "baad", "mein", "se", "ka", "ki", "ko", "toh", "bhi", "batao",
               "boliye", "chahiye", "hoga", "gaya", "kitna", "paisa"}
DEVANAGARI = re.compile(r"[ऀ-ॿ]")


def _pattern(phrase: str) -> re.Pattern:
    # \b = word boundary, so "sure" does not match inside "pressure".
    return re.compile(r"\b" + re.escape(phrase) + r"\b")


def _count(text: str, phrase: str) -> int:
    return len(_pattern(phrase).findall(text))


def _has(text: str, phrases: list[str]) -> bool:
    return any(_count(text, p) for p in phrases)


def detect_language(text: str) -> str:
    if DEVANAGARI.search(text):
        return "Hindi"
    words = re.findall(r"[a-z']+", text.lower())
    if not words:
        return "other"
    share = sum(w in HINDI_WORDS for w in words) / len(words)
    if share > 0.5:
        return "Hindi"
    if share > 0.08:
        return "Hinglish"
    return "English"


def analyze_rules(call: Call) -> CallAnalysis:
    everything = call.as_script().lower()
    customer = call.text_for(Speaker.CUSTOMER).lower()
    # Outcome is usually decided near the end, so look at the last few turns first.
    tail = " ".join(t.text for t in call.turns[-4:]).lower()

    outcome = None
    # If the customer's final words decline or postpone, believe the customer over the agent.
    last_customer = next((t.text.lower() for t in reversed(call.turns) if t.speaker == Speaker.CUSTOMER), "")
    for label in (Outcome.NOT_INTERESTED, Outcome.FOLLOW_UP):
        if _has(last_customer, dict(OUTCOME_RULES)[label]):
            outcome = label
            break
    for label, phrases in ([] if outcome else OUTCOME_RULES):
        if _has(tail, phrases):
            outcome = label
            break
    if outcome is None:
        for label, phrases in OUTCOME_RULES:
            if _has(everything, phrases):
                outcome = label
                break
    m = compute_metrics(call)
    if outcome is None:
        outcome = Outcome.NO_CONNECT if m.customer_turns <= 1 else Outcome.UNRESOLVED

    pos = sum(_count(customer, w) for w in POSITIVE)
    neg = sum(_count(customer, w) for w in NEGATIVE)
    sentiment = Sentiment.POSITIVE if pos > neg else Sentiment.NEGATIVE if neg > pos else Sentiment.NEUTRAL

    objections = [label for label, words in OBJECTION_RULES.items() if _has(customer, words)]

    issues = []
    if m.interruptions >= 2:
        issues.append("talked over customer")
    if m.longest_agent_turn_words > 60:
        issues.append("long monologue")
    if m.agent_talk_share > 0.8 and m.turns > 6:
        issues.append("dominated the conversation")
    if call.turns and call.turns[-1].speaker == Speaker.AGENT and call.turns[-1].text.rstrip().endswith("?"):
        issues.append("ended call abruptly")

    adherence = max(1, min(5, 5 - len(issues) - (1 if outcome == Outcome.UNRESOLVED else 0)))

    first_customer = next((t.text for t in call.turns if t.speaker == Speaker.CUSTOMER), "")
    summary = f"Rules-based guess: outcome {outcome.value.lower().replace('_', ' ')}."
    if first_customer:
        summary += f' Customer opened with: "{first_customer[:80]}".'

    next_action = {
        Outcome.FOLLOW_UP: "Follow up with the customer as promised.",
        Outcome.ESCALATED: "Make sure a human picks up the escalated case.",
        Outcome.UNRESOLVED: "Review the call and contact the customer again.",
        Outcome.NO_CONNECT: "Retry the call at a different time.",
    }.get(outcome, "none")

    return CallAnalysis(
        outcome=outcome,
        customer_sentiment=sentiment,
        objections=objections,
        agent_issues=issues,
        script_adherence=adherence,
        language=detect_language(call.text_for(Speaker.CUSTOMER) or everything),
        summary=summary,
        next_action=next_action,
    )
