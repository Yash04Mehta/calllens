# CallLens: AI call-quality analytics

CallLens reads sales and support phone calls (transcripts or audio recordings) and turns them into a dashboard: how many calls hit their goal, what customers objected to, where agents went wrong, and what each call needs next.

It works on calls made by human agents and by AI voice agents. It also understands the transcript format exported by my voice agent, [Aria](https://github.com/Yash04Mehta/aura-voice-agent).

## The problem it solves

A team making 200 calls a day can't listen to them all. Managers usually spot-check a few, so patterns get missed: an objection that keeps coming up, an agent who talks over people, leads promised a callback that never happens. CallLens reviews every call the same way and adds the results up.

## How it works

```
recordings (.mp3/.wav/...)  ──► Gemini transcribes, labels Agent / Customer
                                        │
transcripts (.txt/.json)  ──────────────┤
                                        ▼
                          parse.py: one format for every source
                                        │
               ┌────────────────────────┴────────────────────────┐
               ▼                                                 ▼
  metrics.py (plain code, no AI)                 llm.py (Gemini, structured JSON)
  duration, talk share, interruptions,           outcome, sentiment, objections,
  customer questions                             agent issues, script adherence,
                                                 summary, next action
               └────────────────────────┬────────────────────────┘
                                        ▼
                     aggregate.py: KPIs and counts  ──►  app.py dashboard + CSV
```

Design decisions:

- **The AI judges and the code counts.** Outcome and sentiment need judgement, so Gemini does those. Talk share and interruptions are arithmetic, so plain Python does them and the numbers can't be hallucinated.
- **Structured output.** Gemini must reply in a fixed JSON schema (a Pydantic model). Labels like `CONVERTED` or `FOLLOW_UP` come from a fixed list, so the charts can count them. Invalid replies are retried once, then reported as errors rather than guessed.
- **Business context is editable.** What counts as a "good call" differs between a coding-bootcamp sales team and a skincare support line. The rubric lives in a text box in the sidebar, and changing it changes how calls are judged.
- **Cache.** Each Gemini analysis is saved by a hash of the transcript and context, so reloading the dashboard costs no API calls (useful on the free tier).
- **Offline mode.** A keyword-based analyzer (English and Hinglish) runs with no API key. It is clearly labelled as a rough baseline and exists for demos, tests, and comparison with the AI.
- **One bad call never breaks the batch.** Errors are collected and shown; the other calls still get analysed.

## Outcomes

| Label | Meaning |
|---|---|
| CONVERTED | Goal achieved: demo booked, sale, appointment |
| RESOLVED | Support issue handled |
| FOLLOW_UP | Interested; a callback or follow-up was agreed |
| NOT_INTERESTED | Declined |
| ESCALATED | Handed to a human or a ticket raised |
| UNRESOLVED | Ended without resolving anything |
| NO_CONNECT | Voicemail, no answer, wrong number |

**Connect rate** = calls that reached a person ÷ all calls. **Success rate** = CONVERTED or RESOLVED ÷ connected calls.

## Running it

Requires Python 3.10+.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate      macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt

copy .env.example .env        # macOS/Linux: cp .env.example .env   then paste your Gemini key

streamlit run app.py          # the dashboard
python -m pytest              # 14 tests, no API key needed
python scripts/analyze.py --mode gemini          # command-line version, writes outputs/calls.csv
python scripts/analyze.py --audio path/to/recordings --mode gemini
```

Without a key, choose **Offline rules** in the sidebar.

## Transcript formats

```text
[00:00] Agent: Hi, this is Riya from SkillPath Academy...
[00:07] Customer: Yes, sure, go ahead.
```

```text
[00:05] CUSTOMER  I want to cancel my order 104.
[00:08] TOOL      get_order_details(104) -> status: processing
[00:09] ARIA      I can see order 104 is still processing...
```

```json
{"turns": [{"speaker": "agent", "text": "Hello...", "start": 0}, {"speaker": "customer", "text": "Hi", "start": 3}]}
```

Speaker names such as Agent/Aria/Bot/Rep and Customer/User/Caller/Lead are recognised. Lines without a speaker are treated as a continuation of the previous turn.

## Sample data

`sample_data/` holds 12 **synthetic** calls written for this project: lead-qualification calls for a fictional coding bootcamp, "SkillPath Academy" (English and Hinglish), and one support call in Aria's format. They cover every outcome, including a deliberately pushy agent and an agent who can't answer questions. None of them are real customer calls.

## Project structure

```
calllens/
├── calllens/
│   ├── schema.py      data shapes (Call, Turn, CallAnalysis...)
│   ├── parse.py       transcript formats → Call
│   ├── metrics.py     counts computed from the transcript
│   ├── llm.py         Gemini: analysis and audio transcription
│   ├── heuristic.py   offline keyword analyzer
│   ├── pipeline.py    runs the analysis with caching and error handling
│   └── aggregate.py   KPIs, tables, label counts
├── app.py             Streamlit dashboard
├── scripts/analyze.py command-line version
├── sample_data/       12 synthetic calls
└── tests/             parser, metrics, rules, pipeline and cache tests
```

## Limitations

- AI judgements can be wrong, especially on short or noisy calls. The drill-down view shows the full transcript next to the verdict so a person can check it.
- Transcription quality depends on audio quality. Speaker labels can be swapped when the agent doesn't speak first.
- Durations are estimated from timestamps. Transcripts without timestamps show no duration.
- The offline rules miss sarcasm, negation ("not too expensive") and anything outside their word lists.
- Calls are sent to Google's Gemini API. Real customer calls should be handled under the company's data-privacy rules.

## Next steps

- Compare Gemini's labels against a few hand-labelled calls to measure accuracy
- Trends over time and per-agent breakdowns
- Push FOLLOW_UP and ESCALATED calls into a CRM or a WhatsApp alert automatically
