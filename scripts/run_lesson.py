"""Generate one or more German lessons with the ADK 2.1 Workflow.

Each run produces a fresh, timestamped ``german_lesson_<timestamp>.md`` file
(existing lessons are never overwritten), running against whatever model
provider is configured (Gemini by default, or LM Studio via MODEL_PROVIDER).

Examples
--------
    # one lesson on whatever topic the news node picks
    uv run python scripts/run_lesson.py

    # five lessons in a row
    uv run python scripts/run_lesson.py --count 5

    # a lesson on a specific theme
    uv run python scripts/run_lesson.py --topic "Sport in Deutschland"

    # three themed lessons
    uv run python scripts/run_lesson.py --count 3 --topic "Umwelt und Klima"
"""

import argparse
import asyncio
import glob
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from google.adk.runners import InMemoryRunner
from google.genai import types

import german_teacher.agent as a

APP = "german_teacher"
USER = "tester"

REQUIRED_STATE_KEYS = {
    "recent_news",
    "base_text",
    "vokabeln",
    "understanding_questions",
    "grammer_rule",
    "writing_assignment",
}


def build_prompt(topic: str | None) -> str:
    """Builds the kickoff message; an optional topic steers the news node."""
    if topic:
        return f"Bitte erstelle eine komplette Deutsch-Lektion zum Thema: {topic}."
    return "Bitte erstelle eine komplette Deutsch-Lektion."


async def generate_one(
    runner: InMemoryRunner, topic: str | None, level: str | None
) -> tuple[bool, str | None]:
    """Runs the workflow once and returns (ok, new_lesson_path)."""
    before = set(glob.glob("german_lesson_*.md"))
    state = {"level": level} if level else {}
    session = await runner.session_service.create_session(
        app_name=APP, user_id=USER, state=state
    )
    message = types.Content(role="user", parts=[types.Part(text=build_prompt(topic))])

    start = time.time()
    async for event in runner.run_async(
        user_id=USER, session_id=session.id, new_message=message
    ):
        author = getattr(event, "author", "?")
        out = getattr(event, "output", None)
        if out is not None:
            print(f"  [{time.time() - start:6.1f}s] {author} -> output: {str(out).replace(chr(10), ' ')[:80]}...")
        elif getattr(event, "content", None) and event.content.parts:
            txt = "".join(p.text or "" for p in event.content.parts)
            if txt.strip():
                print(f"  [{time.time() - start:6.1f}s] {author}: {txt[:80].strip()}...")

    session = await runner.session_service.get_session(
        app_name=APP, user_id=USER, session_id=session.id
    )
    new_files = sorted(set(glob.glob("german_lesson_*.md")) - before)
    ok = bool(new_files) and REQUIRED_STATE_KEYS <= set(session.state.keys())
    return ok, (new_files[-1] if new_files else None)


async def main() -> int:
    parser = argparse.ArgumentParser(description="Generate German lessons.")
    parser.add_argument("--count", type=int, default=1, help="number of lessons to generate")
    parser.add_argument("--topic", type=str, default=None, help="optional theme to steer the lesson")
    parser.add_argument("--level", type=str, default=None, help="CEFR level, e.g. A2, B1, C1 (default: B1/B2)")
    args = parser.parse_args()

    print(f"Provider: {'LM Studio' if a.USING_LM_STUDIO else 'Gemini'}")
    if a.USING_LM_STUDIO:
        print(f"Model:    lm_studio/{a.LM_STUDIO_MODEL} @ {a.LM_STUDIO_API_BASE}")
    if args.topic:
        print(f"Topic:    {args.topic}")
    print(f"Level:    {args.level or a.DEFAULT_LEVEL}")

    runner = InMemoryRunner(agent=a.root_agent, app_name=APP)

    generated: list[str] = []
    for i in range(1, args.count + 1):
        print(f"\n=== Lesson {i}/{args.count} ===")
        ok, path = await generate_one(runner, args.topic, args.level)
        if ok and path:
            generated.append(path)
            print(f"  -> saved {path}")
        else:
            print("  -> INCOMPLETE (missing file or state keys)")

    print(f"\nGenerated {len(generated)}/{args.count} lesson(s):")
    for path in generated:
        print(f"  - {path}")
    return 0 if len(generated) == args.count else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
