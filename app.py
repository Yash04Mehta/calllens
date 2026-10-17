"""CallLens dashboard.  Run:  streamlit run app.py"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from calllens.aggregate import count_labels, kpis, to_frame
from calllens.llm import AUDIO_TYPES, DEFAULT_CONTEXT, transcribe_audio
from calllens.parse import parse_folder, parse_json, parse_text
from calllens.pipeline import analyze_calls

load_dotenv()
st.set_page_config(page_title="CallLens", page_icon="📞", layout="wide")

OUTCOME_COLORS = {
    "CONVERTED": "#2e7d32", "RESOLVED": "#66bb6a", "FOLLOW_UP": "#fbc02d", "ESCALATED": "#fb8c00",
    "UNRESOLVED": "#e53935", "NOT_INTERESTED": "#8d6e63", "NO_CONNECT": "#9e9e9e",
}

# ---------------- Sidebar: choose calls and analyzer ----------------
st.sidebar.title("📞 CallLens")
st.sidebar.caption("Call-quality analytics for sales and support teams")

source = st.sidebar.radio("Calls to analyse", ["Sample calls", "Upload transcripts", "Upload recordings"])
has_key = bool(os.getenv("GEMINI_API_KEY"))
mode_label = st.sidebar.radio(
    "Analyzer",
    ["Gemini AI", "Offline rules (no API key)"],
    index=0 if has_key else 1,
    help="Gemini reads each call and judges outcome, sentiment and objections. "
         "Offline rules use keyword matching: free and instant, but much less accurate.",
)
mode = "gemini" if mode_label.startswith("Gemini") else "rules"
if mode == "gemini" and not has_key:
    st.sidebar.error("GEMINI_API_KEY not found in .env, so switch to offline rules or add the key.")

with st.sidebar.expander("Business context (what a good call looks like)"):
    context = st.text_area("Used by Gemini to judge each call", DEFAULT_CONTEXT, height=220,
                           label_visibility="collapsed")

calls, load_errors = [], []
if source == "Sample calls":
    calls = parse_folder(Path("sample_data"))
elif source == "Upload transcripts":
    files = st.sidebar.file_uploader("Transcripts (.txt or .json)", type=["txt", "json"], accept_multiple_files=True)
    for f in files or []:
        try:
            text = f.getvalue().decode("utf-8")
            stem = Path(f.name).stem
            calls.append(parse_json(json.loads(text), stem, f.name) if f.name.endswith(".json")
                         else parse_text(text, stem, f.name))
        except Exception as exc:
            load_errors.append(f"{f.name}: {exc}")
else:
    files = st.sidebar.file_uploader("Recordings", type=[e.strip(".") for e in AUDIO_TYPES], accept_multiple_files=True)
    if files and not has_key:
        st.sidebar.error("Transcribing audio needs GEMINI_API_KEY.")
    elif files and st.sidebar.button("Transcribe with Gemini"):
        texts = {}
        with st.spinner("Transcribing recordings..."):
            for f in files:
                try:
                    with tempfile.NamedTemporaryFile(suffix=Path(f.name).suffix, delete=False) as tmp:
                        tmp.write(f.getvalue())
                    texts[f.name] = transcribe_audio(Path(tmp.name))
                except Exception as exc:
                    load_errors.append(f"{f.name}: {exc}")
        st.session_state["audio_texts"] = texts
    for name, text in st.session_state.get("audio_texts", {}).items():
        try:
            calls.append(parse_text(text, Path(name).stem, name))
        except Exception as exc:
            load_errors.append(f"{name}: {exc}")

for e in load_errors:
    st.sidebar.warning(e)

st.title("Call quality dashboard")
if not calls:
    st.info("Pick sample calls or upload transcripts/recordings in the sidebar to begin.")
    st.stop()
if mode == "gemini" and not has_key:
    st.stop()


@st.cache_data(show_spinner=False)
def run(scripts: tuple, mode: str, context: str):
    """Cached on the transcripts' text, so re-renders don't repeat API calls."""
    by_id = {c.call_id: c for c in calls}
    return analyze_calls([by_id[cid] for cid, _ in scripts], mode=mode, business_context=context)


with st.spinner(f"Analysing {len(calls)} calls with {'Gemini' if mode == 'gemini' else 'offline rules'}..."):
    reports, errors = run(tuple((c.call_id, c.as_script()) for c in calls), mode, context)
for e in errors:
    st.warning(f"Could not analyse {e}")
if not reports:
    st.stop()

if mode == "rules":
    st.caption("⚠️ Offline rules mode: outcomes and sentiment are keyword guesses. Use Gemini for real judgement.")

df = to_frame(reports)
k = kpis(reports)

# ---------------- KPI cards ----------------
c = st.columns(6)
c[0].metric("Calls", k["calls"])
c[1].metric("Connect rate", f"{k['connect_rate']:.0%}", help="Calls that reached a person")
c[2].metric("Success rate", f"{k['success_rate']:.0%}", help="Connected calls that ended CONVERTED or RESOLVED")
c[3].metric("Avg duration", f"{k['avg_duration_s']:.0f}s" if k["avg_duration_s"] else "n/a")
c[4].metric("Script adherence", f"{k['avg_adherence']:.1f} / 5")
c[5].metric("Need follow-up", k["needs_follow_up"], help="Follow-up, escalated or unresolved calls")

# ---------------- Charts ----------------
left, mid, right = st.columns(3)
with left:
    st.subheader("Outcomes")
    oc = df["outcome"].value_counts().rename_axis("outcome").reset_index(name="calls")
    st.altair_chart(alt.Chart(oc).mark_bar().encode(
        x=alt.X("calls:Q", title=None, axis=alt.Axis(tickMinStep=1, format="d")), y=alt.Y("outcome:N", sort="-x", title=None),
        color=alt.Color("outcome:N", legend=None,
                        scale=alt.Scale(domain=list(OUTCOME_COLORS), range=list(OUTCOME_COLORS.values()))),
        tooltip=["outcome", "calls"]), width="stretch")
with mid:
    st.subheader("Top objections")
    ob = count_labels(reports, "objections")
    if len(ob):
        st.altair_chart(alt.Chart(ob).mark_bar(color="#5c6bc0").encode(
            x=alt.X("calls:Q", title=None, axis=alt.Axis(tickMinStep=1, format="d")), y=alt.Y("label:N", sort="-x", title=None),
            tooltip=["label", "calls"]), width="stretch")
    else:
        st.write("No objections found.")
with right:
    st.subheader("Agent issues")
    ai = count_labels(reports, "agent_issues")
    if len(ai):
        st.altair_chart(alt.Chart(ai).mark_bar(color="#ef6c00").encode(
            x=alt.X("calls:Q", title=None, axis=alt.Axis(tickMinStep=1, format="d")), y=alt.Y("label:N", sort="-x", title=None),
            tooltip=["label", "calls"]), width="stretch")
    else:
        st.write("No agent issues found.")

st.subheader("Sentiment by language")
sl = df.groupby(["language", "sentiment"]).size().reset_index(name="calls")
st.altair_chart(alt.Chart(sl).mark_bar().encode(
    x=alt.X("calls:Q", stack="normalize", title="share of calls"), y=alt.Y("language:N", title=None),
    color=alt.Color("sentiment:N", scale=alt.Scale(domain=["POSITIVE", "NEUTRAL", "NEGATIVE"],
                                                   range=["#43a047", "#bdbdbd", "#e53935"])),
    tooltip=["language", "sentiment", "calls"]).properties(height=140), width="stretch")

# ---------------- Table + download ----------------
st.subheader("All calls")
outcome_filter = st.multiselect("Filter by outcome", sorted(df["outcome"].unique()))
view = df[df["outcome"].isin(outcome_filter)] if outcome_filter else df
st.dataframe(view[["call_id", "outcome", "sentiment", "language", "adherence", "duration_s",
                   "objections", "agent_issues", "next_action"]],
             width="stretch", hide_index=True)
st.download_button("Download CSV", df.to_csv(index=False).encode("utf-8"), "calllens_calls.csv", "text/csv")

# ---------------- Drill-down ----------------
st.subheader("Call detail")
pick = st.selectbox("Choose a call", [r.call.call_id for r in reports])
r = next(x for x in reports if x.call.call_id == pick)
a, m = r.analysis, r.metrics
d1, d2 = st.columns([2, 3])
with d1:
    st.markdown(f"**Outcome:** {a.outcome.value}  \n**Sentiment:** {a.customer_sentiment.value}  \n"
                f"**Language:** {a.language}  \n**Script adherence:** {a.script_adherence}/5")
    st.markdown(f"**Summary:** {a.summary}")
    st.markdown(f"**Next action:** {a.next_action}")
    st.markdown(f"**Objections:** {', '.join(a.objections) or 'none'}  \n"
                f"**Agent issues:** {', '.join(a.agent_issues) or 'none'}")
    st.markdown(f"**Measured:** {m.turns} turns · agent spoke {m.agent_talk_share:.0%} of words · "
                f"{m.interruptions} interruptions · {m.customer_questions} customer questions")
with d2:
    for t in r.call.turns:
        role = {"agent": "assistant", "customer": "user"}.get(t.speaker.value)
        if role is None:
            st.caption(f"⚙️ {t.text}")
            continue
        with st.chat_message(role):
            st.write(t.text + ("  _[interrupted]_" if t.interrupted else ""))
