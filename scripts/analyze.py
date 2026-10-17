"""Command-line version of the dashboard.

    python scripts/analyze.py                         # offline rules on sample_data/
    python scripts/analyze.py --mode gemini           # Gemini (needs GEMINI_API_KEY in .env)
    python scripts/analyze.py --input my_calls/ --context my_business.txt
    python scripts/analyze.py --audio recordings/     # transcribe audio first, then analyse
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv  # noqa: E402

from calllens.aggregate import count_labels, kpis, to_frame  # noqa: E402
from calllens.llm import AUDIO_TYPES, DEFAULT_CONTEXT, transcribe_audio  # noqa: E402
from calllens.parse import parse_folder, parse_text  # noqa: E402
from calllens.pipeline import analyze_calls  # noqa: E402


def main() -> None:
    load_dotenv()
    ap = argparse.ArgumentParser(description="Analyse call transcripts.")
    ap.add_argument("--input", default="sample_data", help="folder of .txt/.json transcripts")
    ap.add_argument("--audio", help="folder of call recordings to transcribe with Gemini first")
    ap.add_argument("--mode", choices=["rules", "gemini"], default="rules")
    ap.add_argument("--context", help="text file describing the business and call goals")
    ap.add_argument("--out", default="outputs/calls.csv")
    args = ap.parse_args()

    context = Path(args.context).read_text(encoding="utf-8") if args.context else DEFAULT_CONTEXT

    if args.audio:
        calls = []
        for f in sorted(Path(args.audio).iterdir()):
            if f.suffix.lower() in AUDIO_TYPES:
                print(f"Transcribing {f.name} ...")
                text = transcribe_audio(f)
                Path("outputs/transcripts").mkdir(parents=True, exist_ok=True)
                Path(f"outputs/transcripts/{f.stem}.txt").write_text(text, encoding="utf-8")
                calls.append(parse_text(text, f.stem, f.name))
    else:
        calls = parse_folder(Path(args.input))

    reports, errors = analyze_calls(calls, mode=args.mode, business_context=context,
                                    on_progress=lambda i, n: print(f"  analysed {i}/{n}", end="\r"))
    print()
    for e in errors:
        print("ERROR", e)

    df = to_frame(reports)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)

    k = kpis(reports)
    if not k["calls"]:
        print("No calls analysed.")
        return
    print(f"\nCalls: {k['calls']}   connect rate: {k['connect_rate']:.0%}   success rate: {k['success_rate']:.0%}"
          f"   avg adherence: {k['avg_adherence']:.1f}/5   need follow-up: {k['needs_follow_up']}")
    print("\nOutcomes:\n" + df["outcome"].value_counts().to_string())
    top = count_labels(reports, "objections")
    if len(top):
        print("\nTop objections:\n" + top.to_string(index=False))
    print(f"\nSaved {args.out}  (analyzer: {args.mode})")


if __name__ == "__main__":
    main()
