import json
from pathlib import Path

import pytest

from calllens.aggregate import count_labels, kpis, to_frame
from calllens.heuristic import analyze_rules, detect_language
from calllens.metrics import compute_metrics
from calllens.parse import parse_folder, parse_json, parse_text
from calllens.pipeline import analyze_calls
from calllens.schema import CallAnalysis, Outcome, Sentiment, Speaker

SAMPLES = Path(__file__).resolve().parents[1] / "sample_data"


def test_parse_chat_style_with_timestamps_and_wrapping():
    call = parse_text("[00:00] Agent: Hello there\nthis line wraps\n[01:05] Customer: Hi?", "c1")
    assert [t.speaker for t in call.turns] == [Speaker.AGENT, Speaker.CUSTOMER]
    assert call.turns[0].text == "Hello there this line wraps"
    assert call.turns[1].start_s == 65


def test_parse_aria_format_with_tool_and_interrupted():
    text = "[00:00] ARIA      Hi from Aura [interrupted]\n[00:03] CUSTOMER  Cancel 104\n[00:04] TOOL      cancel_order(104)"
    call = parse_text(text, "aria")
    assert [t.speaker for t in call.turns] == [Speaker.AGENT, Speaker.CUSTOMER, Speaker.TOOL]
    assert call.turns[0].interrupted and call.turns[0].text == "Hi from Aura"


def test_parse_json_and_rejects_empty():
    call = parse_json({"turns": [{"speaker": "bot", "text": "Hi", "start": 1}, {"speaker": "user", "text": "Yo"}]}, "j")
    assert call.turns[0].speaker == Speaker.AGENT and call.turns[1].start_s is None
    with pytest.raises(ValueError):
        parse_text("no speakers here at all", "bad")


def test_metrics():
    call = parse_text("[00:00] Agent: one two three [interrupted]\n[00:10] Customer: four?", "m")
    m = compute_metrics(call)
    assert m.agent_talk_share == 0.75
    assert m.interruptions == 1 and m.customer_questions == 1
    assert m.duration_s == pytest.approx(10.4)


def test_all_samples_parse():
    calls = parse_folder(SAMPLES)
    assert len(calls) == 12
    assert all(len(c.turns) >= 1 for c in calls)


@pytest.mark.parametrize("name,expected", [
    ("call_01_demo_booked", Outcome.CONVERTED),
    ("call_03_not_interested", Outcome.NOT_INTERESTED),
    ("call_04_pushy_agent", Outcome.FOLLOW_UP),      # customer's last words beat the agent's push
    ("call_05_voicemail", Outcome.NO_CONNECT),
    ("call_10_angry_escalated", Outcome.ESCALATED),
    ("call_12_aria_support", Outcome.RESOLVED),
])
def test_rules_outcomes_on_samples(name, expected):
    call = next(c for c in parse_folder(SAMPLES) if c.call_id == name)
    assert analyze_rules(call).outcome == expected


def test_rules_word_boundaries_and_language():
    call = parse_text("Agent: Shall I book?\nCustomer: This is too much pressure", "p")
    assert analyze_rules(call).customer_sentiment == Sentiment.NEGATIVE   # "sure" inside "pressure" ignored
    assert detect_language("haan theek hai, kal call karo") == "Hindi"
    assert detect_language("yes I need the fees details, kal call karo") == "Hinglish"
    assert detect_language("I would like the details please") == "English"


def fake_analyzer(call):
    return CallAnalysis(outcome=Outcome.CONVERTED if "book" in call.as_script().lower() else Outcome.NO_CONNECT,
                        customer_sentiment=Sentiment.NEUTRAL, objections=["Price "], agent_issues=[],
                        script_adherence=4, language="English", summary="x", next_action="none")


def test_pipeline_kpis_and_error_isolation():
    calls = [parse_text("Agent: want to book?\nCustomer: yes", "a"),
             parse_text("Agent: hello?", "b"),
             parse_text("Agent: x\nCustomer: y", "boom")]

    def flaky(call):
        if call.call_id == "boom":
            raise RuntimeError("api down")
        return fake_analyzer(call)

    reports, errors = analyze_calls(calls, analyzer=flaky, cache_dir=None)
    assert len(reports) == 2 and errors == ["boom: api down"]
    k = kpis(reports)
    assert k["connect_rate"] == 0.5 and k["success_rate"] == 1.0
    assert count_labels(reports, "objections").iloc[0].tolist() == ["price", 2]   # labels normalised
    assert list(to_frame(reports)["call_id"]) == ["a", "b"]


def test_gemini_cache_avoids_repeat_calls(tmp_path, monkeypatch):
    import calllens.llm as llm

    hits = []

    def fake_analyze_call(call, business_context, client=None):
        hits.append(call.call_id)
        return fake_analyzer(call)

    monkeypatch.setattr(llm, "analyze_call", fake_analyze_call)
    monkeypatch.setattr(llm, "_client", lambda: object())
    calls = [parse_text("Agent: book now?\nCustomer: ok", "a")]
    analyze_calls(calls, mode="gemini", business_context="ctx", cache_dir=tmp_path)
    analyze_calls(calls, mode="gemini", business_context="ctx", cache_dir=tmp_path)
    assert hits == ["a"]                                    # second run served from cache
    analyze_calls(calls, mode="gemini", business_context="new ctx", cache_dir=tmp_path)
    assert hits == ["a", "a"]                               # context changed -> re-analysed
    cached = json.loads(next(tmp_path.iterdir()).read_text())
    assert cached["outcome"] == "CONVERTED"
