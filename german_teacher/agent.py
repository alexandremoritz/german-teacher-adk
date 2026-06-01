"""German Learning Exercise Generator — ADK 2.1 Workflow.

The pipeline is expressed as a graph-based :class:`Workflow` (ADK 2.1). The
deprecated ``SequentialAgent`` / ``ParallelAgent`` orchestration of earlier
versions has been replaced by explicit graph edges:

    START → RecentNews → Writer → ┌ Vokabeln  ┐
                                  ├ Verstanden ┤→ Join → SaveLesson
                                  ├ Grammatik  ┤
                                  └ Schreiben  ┘

The LLM steps stay as ``LlmAgent`` instances — ADK runs them as ``single_turn``
workflow nodes, and their ``output_key`` still writes to shared workflow state,
so ``{recent_news}`` / ``{base_text}`` template injection keeps working.

Model selection is configurable so you can run the whole pipeline against a
local LM Studio server instead of Gemini (see ``build_model`` below).
"""

from __future__ import annotations

import json
import os
import random
import re
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime
from typing import Any

from google.adk import Workflow
from google.adk.agents import LlmAgent
from google.adk.agents.context import Context
from google.adk.models.google_llm import Gemini
from google.adk.models.lite_llm import LiteLlm
from google.adk.workflow import JoinNode, START
from google.genai import types

# ---------------------------------------------------------------------------
# Model configuration
# ---------------------------------------------------------------------------
#
# Set MODEL_PROVIDER=lmstudio to run everything against a local LM Studio
# server. Defaults to Gemini for backwards compatibility.
#
#   $env:MODEL_PROVIDER   = "lmstudio"                  # "gemini" (default) | "lmstudio"
#   $env:LM_STUDIO_MODEL  = "qwen/qwen3.6-27b"          # the model id shown in LM Studio
#   $env:LM_STUDIO_API_BASE = "http://localhost:1234/v1"  # LM Studio default
#

retry_config = types.HttpRetryOptions(
    attempts=5,
    exp_base=7,
    initial_delay=1,
    http_status_codes=[429, 500, 503, 504],
)

MODEL_PROVIDER = os.getenv("MODEL_PROVIDER", "gemini").strip().lower()
USING_LM_STUDIO = MODEL_PROVIDER in {"lmstudio", "lm_studio", "lm-studio", "local"}

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite")
LM_STUDIO_MODEL = os.getenv("LM_STUDIO_MODEL", "google/gemma-4-26b-a4b")
LM_STUDIO_API_BASE = os.getenv("LM_STUDIO_API_BASE", "http://localhost:1234/v1")
# LM Studio ignores the key, but LiteLLM/OpenAI clients require a non-empty one.
LM_STUDIO_API_KEY = os.getenv("LM_STUDIO_API_KEY", "lm-studio")


def build_model():
    """Returns a fresh model instance for the configured provider.

    Gemini (the default) talks to Google's API; LM Studio is reached through
    LiteLLM's OpenAI-compatible ``lm_studio/`` provider.
    """
    if USING_LM_STUDIO:
        return LiteLlm(
            model=f"lm_studio/{LM_STUDIO_MODEL}",
            api_base=LM_STUDIO_API_BASE,
            api_key=LM_STUDIO_API_KEY,
        )
    return Gemini(model=GEMINI_MODEL, retry_options=retry_config)


# ---------------------------------------------------------------------------
# News source (a real RSS feed)
# ---------------------------------------------------------------------------
#
# The news comes from a real German news RSS feed (Tagesschau by default), so
# every lesson can link to the original article — and it works on any model
# provider, including local LM Studio models that cannot use Google grounding.
# Override the feed with NEWS_RSS_URL.

NEWS_RSS_URL = os.getenv("NEWS_RSS_URL", "https://www.tagesschau.de/index~rss2.xml")
_RSS_USER_AGENT = "Mozilla/5.0 (german-teacher-adk)"

# CEFR difficulty level. Used by the writer/exercise agents via {level} and can
# be overridden per run through session state (e.g. from the UI).
DEFAULT_LEVEL = os.getenv("LESSON_LEVEL", "B1/B2")


def _input_text(node_input: Any) -> str:
    """Extracts plain text from the workflow's kickoff message (the topic)."""
    if node_input is None:
        return ""
    if isinstance(node_input, str):
        return node_input
    parts = getattr(node_input, "parts", None)
    if parts:
        return " ".join(p.text or "" for p in parts).strip()
    return str(node_input)


def _fetch_rss_articles(url: str) -> list[dict]:
    """Fetches and parses an RSS feed into a list of article dicts."""
    req = urllib.request.Request(url, headers={"User-Agent": _RSS_USER_AGENT})
    with urllib.request.urlopen(req, timeout=15) as resp:
        raw = resp.read()

    root = ET.fromstring(raw)
    articles: list[dict] = []
    for item in root.findall(".//item"):
        title = (item.findtext("title") or "").strip()
        link = (item.findtext("link") or "").strip()
        desc = (item.findtext("description") or "").strip()
        # Keep real articles with a usable teaser; skip video/livestream stubs.
        if title and link.endswith(".html") and len(desc) >= 80:
            articles.append({"title": title, "link": link, "description": desc})
    return articles


def _select_article(articles: list[dict], topic: str) -> dict:
    """Picks an article, preferring ones matching the requested topic."""
    if topic:
        keywords = [w for w in re.findall(r"\w+", topic.lower()) if len(w) > 3]
        matches = [
            a
            for a in articles
            if any(k in (a["title"] + " " + a["description"]).lower() for k in keywords)
        ]
        if matches:
            return random.choice(matches)
    return random.choice(articles)


def fetch_news(ctx: Context, node_input: Any = None) -> str:
    """Fetches a real, recent German news article and writes it to state.

    Sets the ``recent_news`` (headline + teaser), ``news_title`` and
    ``news_url`` state keys. An optional topic in the user's message is used to
    pick a matching article when one is available. Falls back gracefully (no
    link) if the feed is unreachable.

    If ``recent_news`` is already pinned in state, the article is reused as-is
    (no new fetch) — this is how the same news is regenerated at a different
    CEFR level.
    """
    topic = _input_text(node_input)

    # Ensure a CEFR level is always available for downstream {level} templating.
    ctx.state["level"] = ctx.state.get("level") or DEFAULT_LEVEL

    # Reuse a pinned article (same-news level switch / replay): keep the seed
    # and source link, skip the RSS fetch entirely.
    if ctx.state.get("recent_news"):
        ctx.state["news_title"] = ctx.state.get("news_title", "")
        ctx.state["news_url"] = ctx.state.get("news_url", "")
        return f"Reusing pinned article: {ctx.state.get('news_title') or '(pinned)'}"

    # The feed can be overridden per run via state (default: NEWS_RSS_URL).
    feed_url = ctx.state.get("news_rss_url") or NEWS_RSS_URL
    try:
        articles = _fetch_rss_articles(feed_url)
    except Exception:  # noqa: BLE001 - network/parse issues should not abort the run
        articles = []

    if articles:
        article = _select_article(articles, topic)
        ctx.state["news_title"] = article["title"]
        ctx.state["news_url"] = article["link"]
        ctx.state["recent_news"] = f"{article['title']}\n\n{article['description']}"
        return f"Selected article: {article['title']}"

    # Fallback: feed unavailable -> generic seed, no source link.
    ctx.state["news_title"] = ""
    ctx.state["news_url"] = ""
    ctx.state["recent_news"] = topic or "ein aktuelles, interessantes Thema aus Deutschland"
    return "No RSS article available; using a generic topic."


# ---------------------------------------------------------------------------
# LLM agents (run as single_turn workflow nodes)
# ---------------------------------------------------------------------------

LESSON_STYLE_GUIDE = """Shared lesson style guide:
- The learner is one self-learner. Use informal "du" whenever you address the learner.
- Write in German unless an English translation is explicitly requested.
- Output only the content for your own lesson section. Do not add section headings like "## ...".
- Do not include greetings, role descriptions, process notes, or meta-comments about language learning.
- Do not reveal analysis, hidden reasoning, planning notes, or text labeled "thinking process".
- Start immediately with the requested final content. Never start with "Here's", "Plan", "Thinking Process", "Analyse", or similar.
- Use clean Markdown only. Do not output stray asterisks, code fences, HTML, or malformed lists.
- Match CEFR level {level}: control sentence length, abstraction, grammar, vocabulary, and task complexity.
- Stay grounded in the supplied article/text. Do not invent new named facts, statistics, dates, quotes, scores, or outcomes.
- Do not predict future developments or claim a problem is resolved unless the source text says so.
- Avoid future-tense speculation with "wird/werden" unless the source explicitly states that future action.
- Keep wording grammatical and idiomatic. Before final output, silently proofread adjective endings, articles, cases, verb agreement, plural forms, and word order.
- Never use vague filler words such as "Ding", "Sache", "super", "toll", or "spannend". Use precise simple nouns instead.
- Never mix learner address forms. Use "du" for tasks/explanations; do not address the learner with formal "Sie", "Ihnen", or "Ihr/Ihre".
- Never write "du musst" or "du sollst". Use neutral rule language ("Das Verb steht ...") or direct task verbs ("Schreibe ...", "Beschreibe ...").
- Avoid slash pairs such as "Dingen/Leuten". Choose one precise German word.
- In grammar explanations, avoid ambiguous capitalized "Sie" at the start of a sentence. Repeat the noun instead, e.g. "Modalverben zeigen ...".

"""

writer_agent = LlmAgent(
    name="WriterAgent",
    model=build_model(),
    description="Writes a text based on the news.",
    instruction=LESSON_STYLE_GUIDE
    + """You are a careful German teacher writing the reading text for a self-learner.

    Input:
    - Recent News (headline + teaser of a real article): {recent_news}
    - Target CEFR level: {level}

    Task:
    - Write exactly 5 coherent paragraphs in German.
    - Base the text on the provided news item and expand only with plausible context.
    - Do not address the learner directly. Avoid "du", "dein", "Sie", "Ihre", and direct instructions inside the reading text.
    - Do not use headings, bullet points, numbered lists, or direct questions to the learner.
    - Do not end with study advice, motivational language, or a "why this is useful for German learners" paragraph.
    - Do not add new outcomes, predictions, warnings ending, votes succeeding, injuries, causes, or consequences not stated in the source.
    - Do not write speculative future sentences such as "Er wird nun versuchen ..." unless the source says this explicitly.
    - For A1/A2: use short, concrete sentences; explain necessary context simply; avoid abstract political or technical wording where possible.
    - For A1/A2: do a strict grammar check for articles, adjective endings, singular/plural agreement, and verb forms.
    - For B1/B2: use richer connectors and some topic-specific vocabulary, but keep the explanations learner-readable.
    - For C1/C2: allow nuance, argumentation, and more complex sentence structures while staying clear.
    - Keep all five paragraphs focused on the news story itself.
    """,
    output_key="base_text",
)

memo_agent = LlmAgent(
    name="VokabelnAgent",
    model=build_model(),
    description="Extracts difficult words and provides translations.",
    instruction=LESSON_STYLE_GUIDE
    + """You are a vocabulary expert selecting useful words from the reading text.

    Input:
    - Text: {base_text}
    - Target CEFR level: {level}

    Task:
    - Select exactly 5 useful words or short phrases that appear in the text.
    - Choose items that are challenging but realistic for a CEFR {level} learner.
    - Prefer topic-relevant vocabulary over generic words.
    - For nouns, include article and plural in this form: "der Konflikt, -e" or "die Reform, -en". If plural is uncommon, write "kein Plural".
    - Keep each German definition simpler than the reading text level.
    - Definitions must be natural German, not literal English translations.
    - Example sentences must sound natural in German. Avoid awkward collocations such as "Das Alter ist groß"; write a simpler natural sentence instead.
    - Use exactly this one-line Markdown format for every entry, including bold word and literal labels:
      1. **der Konflikt, -e** - Definition: ein Streit zwischen Gruppen - English: conflict - Beispiel: Der Konflikt dauert lange.
    - Each entry is invalid if it does not include all three labels: "Definition:", "English:", and "Beispiel:".
    - Number entries 1 to 5. Do not add any extra notes.
    """,
    output_key="vokabeln",
)

understand_agent = LlmAgent(
    name="VerstandenAgent",
    model=build_model(),
    description="Generates comprehension questions.",
    instruction=LESSON_STYLE_GUIDE
    + """You are a reading comprehension expert creating learner questions.

    Input:
    - Text: {base_text}
    - Target CEFR level: {level}

    Task:
    - Create exactly 3 comprehension questions in German.
    - Number them 1 to 3.
    - Question 1: ask about the main idea.
    - Question 2: ask about one concrete detail from the text.
    - Question 3: ask for a simple inference or opinion that is clearly based on the text.
    - Make every question answerable from the text.
    - Keep each question to one sentence.
    - Avoid multi-part questions, especially for A1, A2, B1, and B2.
    """,
    output_key="understanding_questions",
)

grammar_agent = LlmAgent(
    name="GrammatikAgent",
    model=build_model(),
    description="Explains a grammar rule found in the text.",
    instruction=LESSON_STYLE_GUIDE
    + """You are a German grammar teacher explaining one useful pattern from the reading text.

    Input:
    - Text: {base_text}
    - Target CEFR level: {level}

    Task:
    - Pick exactly one grammar concept that appears in the text and fits CEFR {level}.
    - Choose level-appropriate grammar:
      - A1/A2: word order, modal verbs, present tense, cases, articles, negation with "nicht"/"kein".
      - B1/B2: subordinate clauses, passive voice, relative clauses, connectors, verb placement.
      - C1/C2: nominalization, Konjunktiv I/II, participial constructions, complex sentence style.
    - Quote one complete example sentence verbatim from the text. The quote must appear exactly in the text.
    - Explain the rule in clear German, using simpler language than the reading text.
    - Use precise grammar terms: "konjugiertes Verb", "Infinitiv", "Subjekt", "Objekt", "Artikel", "Kasus". Do not say "Verbstamm" when you mean a verb form.
    - Explain rules as facts, not commands. Do not use "du musst" or "du sollst".
    - Keep the explanation short: 3 to 5 sentences maximum.
    - Use exactly this Markdown format:
      **Konzept:** ...

      **Erklärung:** ...

      **Beispiel aus dem Text:** "..."

      **Mini-Muster:** ...
    """,
    output_key="grammer_rule",
)

writing_assignment_agent = LlmAgent(
    name="SchreibaufgabeAgent",
    model=build_model(),
    description="Creates a writing assignment for the student.",
    instruction=LESSON_STYLE_GUIDE
    + """You are a writing teacher creating one clear learner task.

    Input:
    - Text: {base_text}
    - Target CEFR level: {level}

    Task:
    - Create one writing assignment in German, suitable for CEFR {level}.
    - Address the learner only with "du".
    - Do not use "Sie", "Ihnen", "Ihre", "Schreiben Sie", or "Wählen Sie".
    - Keep the task closely connected to the article topic.
    - Give one clear prompt, not multiple competing options.
    - The prompt must be a task the learner can complete from the reading text and personal opinion; do not require outside knowledge.
    - Include a target length: for A1/A2 use 60-80 words; for B1 and above use about 100 words.
    - Include 3 or 4 guiding bullet points.
    - Do not say "du musst" or "du sollst". Prefer direct task language such as "Schreibe ..." or "Beschreibe ...".
    - Use exactly this Markdown shape:
      **Aufgabe:** ...

      Schreibe ...

      Beachte diese Punkte:
      - ...
      - ...
      - ...
    """,
    output_key="writing_assignment",
)

# ---------------------------------------------------------------------------
# Aggregate + save (a state-bound FunctionNode)
# ---------------------------------------------------------------------------


def save_lesson(
    base_text: str = "",
    vokabeln: str = "",
    understanding_questions: str = "",
    grammer_rule: str = "",
    writing_assignment: str = "",
    news_title: str = "",
    news_url: str = "",
    recent_news: str = "",
    level: str = "",
) -> str:
    """Aggregates the generated exercises into Markdown and saves them to a file.

    Parameters are auto-bound by name from the workflow state, i.e. from the
    ``output_key`` of each upstream agent and from ``fetch_news``.

    Returns:
        A confirmation message followed by the full lesson.
    """
    # Link to the original article right under the lesson title (when available).
    source = f"\n**Quelle:** [{news_title or 'Originalartikel'}]({news_url})\n" if news_url else ""

    # Embed the article seed + level so the same news can be regenerated at a
    # different level later (the comment is invisible in rendered Markdown).
    meta = json.dumps(
        {
            "news_title": news_title,
            "news_url": news_url,
            "recent_news": recent_news,
            "level": level,
        },
        ensure_ascii=False,
    )

    lesson = f"""# German Learning Exercises
{source}
## Text basierend auf aktuellen Nachrichten
{base_text}

## Vokabeln
{vokabeln}

## Verständnisfragen
{understanding_questions}

## Grammatik
{grammer_rule}

## Schreibaufgabe
{writing_assignment}

<!-- lesson-meta: {meta} -->
"""

    filename = f"german_lesson_{datetime.now():%Y%m%d_%H%M%S}.md"
    with open(filename, "w", encoding="utf-8") as f:
        f.write(lesson)

    return f"Saved lesson to {filename}\n\n{lesson}"


# A JoinNode waits for all four parallel agents to finish before the lesson is
# saved. Without it, `save_lesson` would be triggered once per parallel agent
# (a plain node fires on every incoming edge), saving four partial files.
collect_exercises = JoinNode(name="CollectExercises")

# ---------------------------------------------------------------------------
# Workflow graph
# ---------------------------------------------------------------------------

root_agent = Workflow(
    name="GenerateExercisesWorkflow",
    description="Generate complete German exercises based on recent news.",
    edges=[
        # Sequential warm-up: fetch a real article, then write the learner text.
        (START, fetch_news, writer_agent),
        # Fan out: four pedagogical agents work from {base_text} in parallel.
        (
            writer_agent,
            (memo_agent, understand_agent, grammar_agent, writing_assignment_agent),
        ),
        # Fan in: join all four, then aggregate + save once.
        (
            (memo_agent, understand_agent, grammar_agent, writing_assignment_agent),
            collect_exercises,
            save_lesson,
        ),
    ],
)
