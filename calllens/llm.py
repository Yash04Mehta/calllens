"""Gemini calls: (1) analyse a transcript into a CallAnalysis, (2) transcribe a call recording.

Structured output: we pass the CallAnalysis Pydantic model as `response_schema`, so Gemini
must reply with JSON in exactly that shape. Pydantic then validates it. If the reply is
malformed we retry once, then give up for that call instead of inventing data.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from .schema import Call, CallAnalysis

DEFAULT_MODEL = "gemini-3.5-flash-lite"

DEFAULT_CONTEXT = """Business: SkillPath Academy, an online coding bootcamp in India.
Call type: outbound lead-qualification calls to people who filled an enquiry form.
Goal of every call: confirm interest, ask about background and budget, handle objections,
and book a free demo class (that counts as CONVERTED).
The agent should: introduce themselves, ask permission to talk, ask at least one qualifying
question, never pressure the customer, and offer a WhatsApp follow-up if the customer is busy."""

SYSTEM = """You are a call-quality analyst. You read one phone call transcript and fill in
the requested JSON fields. Rules:
- Judge only from the transcript. Never invent facts that are not in it.
- AGENT is the company's representative (human or AI). CUSTOMER is the person called. TOOL lines are
  system actions the agent triggered (lookups, cancellations); use them as evidence of what happened.
- Use short lowercase labels for objections and agent issues; empty lists if there are none.
- script_adherence: compare the agent's behaviour to the business context's goals.
- summary and next_action: plain English, even if the call was in Hindi or Hinglish."""


def model_name() -> str:
    return os.getenv("CALLLENS_MODEL", DEFAULT_MODEL)


def _client():
    from google import genai  # imported lazily so the offline mode works without the package configured

    key = os.getenv("GEMINI_API_KEY")
    if not key:
        raise RuntimeError("GEMINI_API_KEY is not set. Put it in a .env file or use the offline rules analyzer.")
    return genai.Client(api_key=key)


def analyze_call(call: Call, business_context: str = DEFAULT_CONTEXT, client=None) -> CallAnalysis:
    from google.genai import types

    client = client or _client()
    prompt = f"BUSINESS CONTEXT:\n{business_context.strip()}\n\nTRANSCRIPT ({call.call_id}):\n{call.as_script()}"
    config = types.GenerateContentConfig(
        system_instruction=SYSTEM,
        response_mime_type="application/json",
        response_schema=CallAnalysis,
        temperature=0.1,   # low = consistent labels from run to run
    )
    last_error: Exception | None = None
    for _ in range(2):
        resp = client.models.generate_content(model=model_name(), contents=prompt, config=config)
        try:
            return CallAnalysis.model_validate(json.loads(resp.text))
        except Exception as exc:  # malformed JSON or a value outside the schema
            last_error = exc
    raise ValueError(f"Gemini returned an invalid analysis for {call.call_id}: {last_error}")


TRANSCRIBE_PROMPT = """Transcribe this phone call. Output one line per speaker turn in exactly this format:
[mm:ss] Agent: <words>
[mm:ss] Customer: <words>
The agent is the company representative who usually speaks first and introduces the company.
Keep Hindi or Hinglish as spoken, written in Latin letters. Do not summarise, do not add anything else."""

AUDIO_TYPES = {".wav": "audio/wav", ".mp3": "audio/mp3", ".m4a": "audio/mp4", ".ogg": "audio/ogg",
               ".flac": "audio/flac", ".aac": "audio/aac", ".webm": "audio/webm"}
INLINE_LIMIT = 18 * 1024 * 1024   # Gemini accepts ~20 MB inline; larger files go through the Files API


def transcribe_audio(path: Path, client=None) -> str:
    """Return a transcript in CallLens's text format, ready for parse.parse_text."""
    from google.genai import types

    path = Path(path)
    mime = AUDIO_TYPES.get(path.suffix.lower())
    if mime is None:
        raise ValueError(f"Unsupported audio type {path.suffix}. Use one of: {', '.join(AUDIO_TYPES)}")
    client = client or _client()
    if path.stat().st_size <= INLINE_LIMIT:
        audio = types.Part.from_bytes(data=path.read_bytes(), mime_type=mime)
    else:
        audio = client.files.upload(file=str(path))
    resp = client.models.generate_content(
        model=model_name(), contents=[TRANSCRIBE_PROMPT, audio],
        config=types.GenerateContentConfig(temperature=0.0),
    )
    return resp.text.strip()
