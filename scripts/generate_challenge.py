#!/usr/bin/env python3
"""Generate one backend engineering challenge with the Gemini API.

Writes `challenges/YYYY-MM-DD.md`. Never writes a solution — the whole point is
that the answer is yours.

    export GEMINI_API_KEY=...
    python scripts/generate_challenge.py                 # today's topic
    python scripts/generate_challenge.py --topic Redis   # override
    python scripts/generate_challenge.py --dry-run       # print, write nothing
    python scripts/generate_challenge.py --list-models   # what your key supports

Standard library only — no pip install in CI, nothing to keep updated.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import urllib.error
import urllib.request
from datetime import datetime
from zoneinfo import ZoneInfo

ROOT = pathlib.Path(__file__).resolve().parent.parent
CHALLENGES = ROOT / "challenges"
PROMPT_FILE = ROOT / "prompts" / "challenge.md"

# Chennai — so "today" matches the day you actually sit down to solve it,
# rather than whatever UTC thinks at 03:30.
TIMEZONE = ZoneInfo("Asia/Kolkata")

API_ROOT = "https://generativelanguage.googleapis.com/v1beta"

# Model id is configurable because Google retires and renames these. If the call
# 404s, the available models are printed automatically.
#
# `or` rather than a get() default: an unset GitHub Actions variable arrives as
# an empty string, not an absent key, and "" would otherwise win over the default.
DEFAULT_MODEL = os.environ.get("GEMINI_MODEL") or "gemini-2.5-flash"

# Monday-first, matching datetime.weekday().
WEEKLY_TOPICS = {
    0: "PostgreSQL",
    1: "TypeScript and Node.js",
    2: "Redis and caching",
    3: "System design",
    4: "AWS and cloud infrastructure",
    5: "Data structures and algorithms",
    6: "Distributed systems",
}


def topic_for(day: datetime) -> str:
    return WEEKLY_TOPICS[day.weekday()]


def load_prompt(topic: str, weekday: str) -> str:
    if not PROMPT_FILE.exists():
        raise SystemExit(f"prompt template missing: {PROMPT_FILE}")
    return PROMPT_FILE.read_text().replace("{{TOPIC}}", topic).replace(
        "{{WEEKDAY}}", weekday
    )


def call_gemini(prompt: str, model: str, api_key: str) -> str:
    url = f"{API_ROOT}/models/{model}:generateContent?key={api_key}"
    body = json.dumps(
        {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                # Some variety day to day, but not so much that it drifts
                # off-topic or starts inventing formats.
                "temperature": 0.9,
                "maxOutputTokens": 2048,
            },
        }
    ).encode()

    request = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/json"}
    )

    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            payload = json.load(response)
    except urllib.error.HTTPError as error:
        detail = error.read().decode(errors="replace")[:600]
        if error.code == 404:
            print(
                f"model {model!r} not found (HTTP 404). Models this key can use:",
                file=sys.stderr,
            )
            try:
                list_models(api_key)
            except SystemExit:
                print("  (could not list models)", file=sys.stderr)
            raise SystemExit(
                "Set the GEMINI_MODEL repository variable to one of the ids above."
            ) from error
        if error.code in (401, 403):
            raise SystemExit(
                f"auth failed (HTTP {error.code}). Check GEMINI_API_KEY.\n\n{detail}"
            ) from error
        raise SystemExit(f"Gemini API error {error.code}:\n{detail}") from error
    except urllib.error.URLError as error:
        raise SystemExit(f"could not reach the Gemini API: {error.reason}") from error

    candidates = payload.get("candidates") or []
    if not candidates:
        # Usually means the prompt tripped a safety filter.
        raise SystemExit(f"no candidates returned:\n{json.dumps(payload, indent=2)[:800]}")

    parts = candidates[0].get("content", {}).get("parts") or []
    text = "".join(part.get("text", "") for part in parts).strip()

    if not text:
        finish = candidates[0].get("finishReason", "unknown")
        raise SystemExit(f"empty response (finishReason={finish})")

    return text


def list_models(api_key: str) -> int:
    url = f"{API_ROOT}/models?key={api_key}"
    try:
        with urllib.request.urlopen(url, timeout=30) as response:
            payload = json.load(response)
    except urllib.error.HTTPError as error:
        raise SystemExit(
            f"HTTP {error.code}: {error.read().decode(errors='replace')[:400]}"
        ) from error

    for model in payload.get("models", []):
        methods = model.get("supportedGenerationMethods", [])
        if "generateContent" in methods:
            name = model["name"].removeprefix("models/")
            print(f"  {name:<40} {model.get('displayName', '')}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--topic", help="override the weekday topic")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--date", help="YYYY-MM-DD, defaults to today in Asia/Kolkata")
    parser.add_argument("--dry-run", action="store_true", help="print, write nothing")
    parser.add_argument("--list-models", action="store_true")
    parser.add_argument(
        "--force", action="store_true", help="overwrite an existing challenge"
    )
    args = parser.parse_args()

    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise SystemExit("GEMINI_API_KEY is not set")

    if args.list_models:
        return list_models(api_key)

    if args.date:
        day = datetime.strptime(args.date, "%Y-%m-%d").replace(tzinfo=TIMEZONE)
    else:
        day = datetime.now(TIMEZONE)

    date_str = day.strftime("%Y-%m-%d")
    weekday = day.strftime("%A")
    topic = args.topic or topic_for(day)

    target = CHALLENGES / f"{date_str}.md"
    if target.exists() and not args.force:
        print(f"{target.relative_to(ROOT)} already exists — nothing to do")
        return 0

    print(f"generating: {date_str} ({weekday}) — {topic}", file=sys.stderr)
    body = call_gemini(load_prompt(topic, weekday), args.model, api_key)

    document = (
        f"# {date_str} — {topic}\n\n"
        f"*{weekday}. Generated as a prompt only; the solution is not included.*\n\n"
        f"---\n\n"
        f"{body}\n\n"
        f"---\n\n"
        f"## Your solution\n\n"
        f"Write it in `solutions/{date_str}/`, then fill this in:\n\n"
        f"**Approach**\n\n_TODO_\n\n"
        f"**What I learned**\n\n_TODO_\n\n"
        f"**What I'd do differently at scale**\n\n_TODO_\n"
    )

    if args.dry_run:
        print(document)
        return 0

    CHALLENGES.mkdir(exist_ok=True)
    target.write_text(document)
    print(f"wrote {target.relative_to(ROOT)}")

    # Surfaced so the workflow can use it in the branch and PR title.
    if github_output := os.environ.get("GITHUB_OUTPUT"):
        with open(github_output, "a") as handle:
            handle.write(f"date={date_str}\n")
            handle.write(f"topic={topic}\n")
            handle.write(f"path=challenges/{date_str}.md\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
