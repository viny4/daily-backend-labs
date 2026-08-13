#!/usr/bin/env python3
"""Generate one backend engineering challenge, and its answer, with the Gemini API.

Writes `challenges/YYYY-MM-DD.md` — the question followed by a worked answer, so
the two can be read together.

    export GEMINI_API_KEY=...
    python scripts/generate_challenge.py                 # today's topic
    python scripts/generate_challenge.py --topic Redis   # override
    python scripts/generate_challenge.py --dry-run       # print, write nothing
    python scripts/generate_challenge.py --list-models   # what your key supports

Answering a challenge that already exists — used to backfill the days that were
generated before answers were part of the flow:

    python scripts/generate_challenge.py --answer --date 2026-08-10

Standard library only — no pip install in CI, nothing to keep updated.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import subprocess
import sys
import urllib.error
import urllib.request
from datetime import datetime
from zoneinfo import ZoneInfo

ROOT = pathlib.Path(__file__).resolve().parent.parent
CHALLENGES = ROOT / "challenges"
CHALLENGE_PROMPT = ROOT / "prompts" / "challenge.md"
SOLUTION_PROMPT = ROOT / "prompts" / "solution.md"

# The heading the answer is written under, plus the placeholder block used
# before answers were generated. Both are recognised so a file can be re-answered
# without stacking up duplicate sections.
ANSWER_HEADING = "## Worked answer"
LEGACY_HEADINGS = ("## Your solution",)

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


def git(*args: str) -> str:
    """Run git in the repo and return stdout, or "" if it fails for any reason.

    Only ever used to enrich the "do not repeat" history, so every failure mode
    — no git, no refs fetched, detached checkout — degrades to a shorter list
    rather than a failed run. A missed duplicate beats no challenge at all.
    """
    try:
        result = subprocess.run(
            ["git", *args], cwd=ROOT, capture_output=True, text=True, timeout=30
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    return result.stdout if result.returncode == 0 else ""


def unmerged_challenges() -> dict[str, str]:
    """Challenge files that exist only on an unmerged `challenge/*` branch.

    A challenge is only on disk once its PR is merged. Skipped days stay on
    their own branch forever, so a history built from the working tree alone
    goes stale the moment you leave a PR open — which the whole design expects
    you to do. Those are exactly the problems most likely to be reissued: the
    unsolved ones.

    Requires the refs to have been fetched; see the fetch step in the workflow.
    """
    found: dict[str, str] = {}

    refs = git("for-each-ref", "--format=%(refname)", "refs/remotes/origin/challenge/")
    for ref in refs.split():
        date = ref.rsplit("/", 1)[-1]

        # Merged already — the working tree copy is authoritative.
        if (CHALLENGES / f"{date}.md").exists():
            continue

        if blob := git("show", f"{ref}:challenges/{date}.md"):
            found[date] = blob

    return found


def summarise(text: str, fallback_title: str) -> str:
    """One line identifying a challenge: its title, plus enough of the scenario
    to recognise the problem without spending the prompt budget on history."""
    lines = text.splitlines()

    title = next(
        (ln.lstrip("# ").strip() for ln in lines if ln.startswith("# ")), fallback_title
    )

    scenario = ""
    for index, line in enumerate(lines):
        if line.strip().lower().startswith("## scenario"):
            for following in lines[index + 1 :]:
                if following.strip():
                    scenario = following.strip()
                    break
            break

    return f"- {title}: {scenario[:180]}" if scenario else f"- {title}"


def recent_challenges(limit: int = 40) -> list[str]:
    """One-line summaries of the most recent challenges, merged or still open.

    The model has no memory between runs, so without this it happily reissues
    "the invoices table got slow, design the indexes" every few Mondays. Feeding
    the recent history back in is the only thing that makes the series varied.
    """
    sources: dict[str, str] = {}

    if CHALLENGES.is_dir():
        for path in CHALLENGES.glob("*.md"):
            sources[path.stem] = path.read_text()

    sources.update(unmerged_challenges())

    # Keys are ISO dates, so a plain reverse sort is newest-first.
    return [
        summarise(sources[date], date)
        for date in sorted(sources, reverse=True)[:limit]
    ]


def challenge_prompt(topic: str, weekday: str) -> str:
    if not CHALLENGE_PROMPT.exists():
        raise SystemExit(f"prompt template missing: {CHALLENGE_PROMPT}")

    history = recent_challenges()
    recent_block = (
        "\n".join(history)
        if history
        else "(none yet — this is the first challenge in the series)"
    )

    return (
        CHALLENGE_PROMPT.read_text()
        .replace("{{TOPIC}}", topic)
        .replace("{{WEEKDAY}}", weekday)
        .replace("{{RECENT}}", recent_block)
    )


def solution_prompt(topic: str, challenge: str) -> str:
    if not SOLUTION_PROMPT.exists():
        raise SystemExit(f"prompt template missing: {SOLUTION_PROMPT}")

    return (
        SOLUTION_PROMPT.read_text()
        .replace("{{TOPIC}}", topic)
        .replace("{{CHALLENGE}}", challenge)
    )


def question_part(document: str) -> str:
    """The question half of a challenge file, with any answer block removed.

    Lets a file be re-answered — or a pre-answers file be backfilled — without
    ending up with two answers stacked on top of each other.
    """
    body = document
    for heading in (ANSWER_HEADING, *LEGACY_HEADINGS):
        index = body.find(f"\n{heading}")
        if index != -1:
            body = body[:index]

    # Drop the trailing separator that sat between question and answer.
    lines = body.rstrip().splitlines()
    while lines and lines[-1].strip() in ("", "---"):
        lines.pop()

    return "\n".join(lines)


def render(question: str, solution: str) -> str:
    return (
        f"{question.rstrip()}\n\n"
        f"---\n\n"
        f"{ANSWER_HEADING}\n\n"
        f"{solution.strip()}\n"
    )


def call_gemini(
    prompt: str,
    model: str,
    api_key: str,
    temperature: float = 0.9,
    max_output_tokens: int = 4096,
) -> str:
    url = f"{API_ROOT}/models/{model}:generateContent?key={api_key}"
    body = json.dumps(
        {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                # Variety day to day for questions; answers run colder, because
                # there a surprising choice is usually just a wrong one.
                "temperature": temperature,
                # 2.5-era models spend output budget on internal reasoning, so
                # a modest cap silently truncates the visible answer. Neither
                # call needs deliberation, so switch thinking off and give the
                # text room. Answers get more of it — they carry the code.
                "maxOutputTokens": max_output_tokens,
                "thinkingConfig": {"thinkingBudget": 0},
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

    finish = candidates[0].get("finishReason", "")
    parts = candidates[0].get("content", {}).get("parts") or []
    text = "".join(part.get("text", "") for part in parts).strip()

    if not text:
        raise SystemExit(f"empty response (finishReason={finish})")

    # Never write a half-finished challenge. A truncated scenario reads as a
    # broken repository, and the failure is silent otherwise.
    if finish == "MAX_TOKENS":
        raise SystemExit(
            "response hit the output limit and would be truncated mid-sentence.\n"
            "Raise maxOutputTokens, or confirm thinkingBudget is 0 for this model."
        )

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
    parser.add_argument(
        "--answer",
        action="store_true",
        help="answer an existing challenge instead of generating a new one",
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

    if args.answer:
        # Backfill: the question already exists, only the answer is missing.
        if not target.exists():
            raise SystemExit(f"no challenge to answer at {target.relative_to(ROOT)}")

        question = question_part(target.read_text())
        if not question.strip():
            raise SystemExit(f"{target.relative_to(ROOT)} has no question in it")

        # The topic is in the title of the file that already exists; the
        # weekday default would be wrong for a backdated file anyway.
        title = question.splitlines()[0].lstrip("# ").strip()
        topic = args.topic or (title.split("—", 1)[-1].strip() if "—" in title else topic)

        print(f"answering: {date_str} — {topic}", file=sys.stderr)
    else:
        if target.exists() and not args.force:
            print(f"{target.relative_to(ROOT)} already exists — nothing to do")
            return 0

        history_count = len(recent_challenges())
        print(
            f"generating: {date_str} ({weekday}) — {topic} "
            f"[avoiding {history_count} previous]",
            file=sys.stderr,
        )
        body = call_gemini(challenge_prompt(topic, weekday), args.model, api_key)

        question = (
            f"# {date_str} — {topic}\n\n"
            f"*{weekday}. Question and worked answer, both generated.*\n\n"
            f"---\n\n"
            f"{body}"
        )

    print(f"writing the answer for {date_str}", file=sys.stderr)
    solution = call_gemini(
        solution_prompt(topic, question),
        args.model,
        api_key,
        # The answer is a reference document, not a brainstorm: run it colder,
        # and give it room for the code it has to carry.
        #
        # 16k rather than 8k because a system design answer covers four
        # deliverables plus the "going further" question and overran 8k — the
        # truncation guard caught it, but a failed run is still a lost day.
        temperature=0.4,
        max_output_tokens=16384,
    )

    document = render(question, solution)

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
