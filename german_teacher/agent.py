"""Generator für Deutschübungen — ADK-2.1-Workflow.

Die Pipeline ist als graphbasierter :class:`Workflow` (ADK 2.1) beschrieben.
Die veraltete Orchestrierung über ``SequentialAgent`` / ``ParallelAgent`` aus
früheren Versionen wurde durch explizite Graph-Kanten ersetzt:

    START → RecentNews → Writer → ┌ Vokabeln  ┐
                                  ├ Verstanden ┤→ Join → SaveLesson
                                  ├ Grammatik  ┤
                                  └ Schreiben  ┘

Die LLM-Schritte bleiben ``LlmAgent``-Instanzen. ADK führt sie als
``single_turn``-Workflow-Knoten aus, und ihr ``output_key`` schreibt weiterhin
in den gemeinsamen Workflow-State. Dadurch funktioniert die Template-Injektion
über ``{recent_news}`` / ``{base_text}`` weiter.

Die Modellauswahl ist konfigurierbar, damit die gesamte Pipeline statt Gemini
auch einen lokalen LM-Studio-Server verwenden kann (siehe ``build_model``).
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
# Modellkonfiguration
# ---------------------------------------------------------------------------
#
# Setze MODEL_PROVIDER=lmstudio, um alles gegen einen lokalen LM-Studio-Server
# laufen zu lassen. Aus Gründen der Rückwärtskompatibilität ist Gemini der
# Standard.
#
#   $env:MODEL_PROVIDER   = "lmstudio"                  # "gemini" (Standard) | "lmstudio"
#   $env:LM_STUDIO_MODEL  = "qwen/qwen3.6-27b"          # Modell-ID aus LM Studio
#   $env:LM_STUDIO_API_BASE = "http://localhost:1234/v1"  # LM-Studio-Standard
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
# LM Studio ignoriert den Key, aber LiteLLM/OpenAI-Clients verlangen einen
# nicht leeren Wert.
LM_STUDIO_API_KEY = os.getenv("LM_STUDIO_API_KEY", "lm-studio")


def build_model():
    """Gibt eine frische Modellinstanz für den konfigurierten Provider zurück.

    Gemini (Standard) spricht mit Googles API. LM Studio wird über den
    OpenAI-kompatiblen ``lm_studio/``-Provider von LiteLLM erreicht.
    """
    if USING_LM_STUDIO:
        return LiteLlm(
            model=f"lm_studio/{LM_STUDIO_MODEL}",
            api_base=LM_STUDIO_API_BASE,
            api_key=LM_STUDIO_API_KEY,
        )
    return Gemini(model=GEMINI_MODEL, retry_options=retry_config)


# ---------------------------------------------------------------------------
# Nachrichtenquelle (ein echter RSS-Feed)
# ---------------------------------------------------------------------------
#
# Die Nachricht kommt aus einem echten deutschen RSS-Feed (standardmäßig
# Tagesschau). So kann jede Lektion auf den Originalartikel verlinken. Das
# funktioniert mit jedem Modell-Provider, auch mit lokalen LM-Studio-Modellen
# ohne Google Grounding. Der Feed kann über NEWS_RSS_URL überschrieben werden.

NEWS_RSS_URL = os.getenv("NEWS_RSS_URL", "https://www.tagesschau.de/index~rss2.xml")
_RSS_USER_AGENT = "Mozilla/5.0 (german-teacher-adk)"

# GER-Schwierigkeitsniveau. Wird von den Text- und Übungsagenten über {level}
# genutzt und kann pro Lauf über den Session-State überschrieben werden, z. B.
# aus der UI.
DEFAULT_LEVEL = os.getenv("LESSON_LEVEL", "B1/B2")


def _input_text(node_input: Any) -> str:
    """Extrahiert Klartext aus der Startnachricht des Workflows (das Thema)."""
    if node_input is None:
        return ""
    if isinstance(node_input, str):
        return node_input
    parts = getattr(node_input, "parts", None)
    if parts:
        return " ".join(p.text or "" for p in parts).strip()
    return str(node_input)


def _fetch_rss_articles(url: str) -> list[dict]:
    """Lädt und parst einen RSS-Feed in eine Liste von Artikel-Dicts."""
    req = urllib.request.Request(url, headers={"User-Agent": _RSS_USER_AGENT})
    with urllib.request.urlopen(req, timeout=15) as resp:
        raw = resp.read()

    root = ET.fromstring(raw)
    articles: list[dict] = []
    for item in root.findall(".//item"):
        title = (item.findtext("title") or "").strip()
        link = (item.findtext("link") or "").strip()
        desc = (item.findtext("description") or "").strip()
        # Nur echte Artikel mit brauchbarem Teaser behalten; Video- und
        # Livestream-Stubs überspringen.
        if title and link.endswith(".html") and len(desc) >= 80:
            articles.append({"title": title, "link": link, "description": desc})
    return articles


def _select_article(articles: list[dict], topic: str) -> dict:
    """Wählt einen Artikel aus und bevorzugt Treffer zum gewünschten Thema."""
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
    """Lädt eine echte, aktuelle deutsche Nachricht und schreibt sie in den State.

    Setzt die State-Keys ``recent_news`` (Überschrift + Teaser),
    ``news_title`` und ``news_url``. Ein optionales Thema aus der Nutzernachricht
    wird genutzt, um nach Möglichkeit einen passenden Artikel auszuwählen. Wenn
    der Feed nicht erreichbar ist, wird sauber auf ein generisches Thema ohne
    Link zurückgegriffen.

    Wenn ``recent_news`` bereits im State fixiert ist, wird der Artikel
    unverändert wiederverwendet. So kann dieselbe Nachricht auf einem anderen
    GER-Niveau neu generiert werden.
    """
    topic = _input_text(node_input)

    # Sicherstellen, dass für nachgelagerte {level}-Templates immer ein
    # GER-Niveau verfügbar ist.
    ctx.state["level"] = ctx.state.get("level") or DEFAULT_LEVEL

    # Fixierten Artikel wiederverwenden (Levelwechsel / Replay derselben
    # Nachricht): Seed und Quellenlink behalten, RSS-Abruf überspringen.
    if ctx.state.get("recent_news"):
        ctx.state["news_title"] = ctx.state.get("news_title", "")
        ctx.state["news_url"] = ctx.state.get("news_url", "")
        return f"Fixierter Artikel wird wiederverwendet: {ctx.state.get('news_title') or '(fixiert)'}"

    # Der Feed kann pro Lauf über den State überschrieben werden
    # (Standard: NEWS_RSS_URL).
    feed_url = ctx.state.get("news_rss_url") or NEWS_RSS_URL
    try:
        articles = _fetch_rss_articles(feed_url)
    except Exception:  # noqa: BLE001 - Netzwerk-/Parsefehler sollen den Lauf nicht abbrechen
        articles = []

    if articles:
        article = _select_article(articles, topic)
        ctx.state["news_title"] = article["title"]
        ctx.state["news_url"] = article["link"]
        ctx.state["recent_news"] = f"{article['title']}\n\n{article['description']}"
        return f"Ausgewählter Artikel: {article['title']}"

    # Fallback: Feed nicht verfügbar -> generischer Seed, kein Quellenlink.
    ctx.state["news_title"] = ""
    ctx.state["news_url"] = ""
    ctx.state["recent_news"] = topic or "ein aktuelles, interessantes Thema aus Deutschland"
    return "Kein RSS-Artikel verfügbar; generisches Thema wird verwendet."


# ---------------------------------------------------------------------------
# LLM-Agenten (laufen als single_turn-Workflow-Knoten)
# ---------------------------------------------------------------------------

LESSON_STYLE_GUIDE = """Gemeinsamer Stil- und Qualitätsrahmen:
- Die lernende Person lernt allein. Sprich sie in Aufgaben und Erklärungen konsequent mit "du" an.
- Schreibe auf Deutsch, außer wenn ausdrücklich eine englische Übersetzung verlangt wird.
- Liefere nur den Inhalt deines eigenen Abschnitts. Schreibe keine Abschnittsüberschriften wie "## ...".
- Beginne sofort mit dem fertigen Inhalt. Keine Begrüßung, keine Rollenbeschreibung, keine Prozessnotiz.
- Keine Metakommentare über das Deutschlernen, keine Lerntipps im Lesetext, keine Motivationsfloskeln.
- Zeige niemals Analyse, versteckte Gedankengänge, Planungsnotizen oder Text mit Labels wie "thinking process", "Plan" oder "Analyse".
- Verwende sauberes Markdown. Keine Codeblöcke, kein HTML, keine losen Sternchen, keine kaputten Listen.
- Halte das Niveau {level} streng ein: Satzlänge, Abstraktion, Grammatik, Wortschatz und Aufgabenkomplexität müssen dazu passen.
- Bleibe beim gelieferten Artikel oder Text. Erfinde keine neuen Namen, Zahlen, Daten, Zitate, Ergebnisse, Ursachen oder Folgen.
- Wenn der Nachrichtenteaser mit "Von <Name>" endet, ist das normalerweise die Autorin oder der Autor des Artikels. Schreibe nicht, dass diese Person ein Gutachten, eine Studie oder ein Ereignis verfasst hat, außer der Text sagt das ausdrücklich.
- Schreibe ganze deutsche Sätze ohne fremdsprachliche Fragmente. Ersetze englische Wörter wie "incident" durch natürliches Deutsch wie "Vorfall".
- Prüfe besonders, dass keine einzelnen Wörter aus anderen Sprachen in Fragen oder Erklärungen stehen.
- Spekuliere nicht über die Zukunft und behaupte nicht, ein Problem sei gelöst, wenn die Quelle das nicht sagt.
- Verwende Zukunftsformen mit "wird/werden" nur, wenn die Quelle eine konkrete zukünftige Handlung nennt.
- Formuliere idiomatisch und grammatisch korrekt. Prüfe vor der Ausgabe still Artikel, Kasus, Adjektivendungen, Verbformen, Kongruenz, Pluralformen und Wortstellung.
- Vermeide vage Füllwörter wie "Ding", "Sache", "super", "toll" oder "spannend". Wähle einfache, präzise Nomen.
- Vermische nie "du" und "Sie". Verwende in Aufgaben und Erklärungen nur "du"; nie "Sie", "Ihnen", "Ihr" oder "Ihre" als Anrede.
- Schreibe nie "du musst" oder "du sollst". Nutze neutrale Regelsprache ("Das Verb steht ...") oder direkte Aufgabenverben ("Schreibe ...", "Beschreibe ...").
- Vermeide Schrägstrich-Paare wie "Dingen/Leuten". Wähle ein passendes deutsches Wort.
- In Grammatikerklärungen darf ein Satz nicht mehrdeutig mit großem "Sie" beginnen. Wiederhole das Nomen, z. B. "Modalverben zeigen ...".
- Auch Beispiel- und Mustersätze dürfen nicht mit "Sie" beginnen. Nutze Namen, "eine Person" oder ein konkretes Nomen.

Harte Ausgaberegeln:
- Gib nur den fertigen Inhalt aus, der zu deiner Aufgabe gehört.
- Keine versteckten Gedankengänge, keine Planung, keine Selbstbewertung, keine Alternativentwürfe.
- Halte exakte Mengenangaben und Markdown-Vorlagen wörtlich ein.
- Lasse in der fertigen Ausgabe keine Platzhalter-Auslassungspunkte ("...") stehen.

"""

writer_agent = LlmAgent(
    name="WriterAgent",
    model=build_model(),
    description="Schreibt einen Lesetext auf Basis der Nachricht.",
    instruction=LESSON_STYLE_GUIDE
    + """Du bist eine sorgfältige Deutschlehrkraft und schreibst den Lesetext für eine selbstlernende Person.

    Eingabe:
    - Aktuelle Nachricht (Überschrift + Teaser eines echten Artikels): {recent_news}
    - Zielniveau nach GER: {level}

    Aufgabe:
    - Schreibe genau 5 zusammenhängende Absätze auf Deutsch.
    - Nutze die gelieferte Nachricht als Grundlage und ergänze nur plausiblen Kontext, der eng daran anschließt.
    - Kopiere den Nachrichtenteaser nicht als eigenen Absatz. Formuliere alle Absätze selbstständig und vermeide Wiederholungen.
    - Sprich die lernende Person im Lesetext nicht direkt an. Kein "du", kein "dein", kein "Sie", kein "Ihre", keine Arbeitsanweisungen.
    - Keine Überschriften, keine Stichpunkte, keine nummerierten Listen und keine direkten Fragen an die lernende Person.
    - Der letzte Absatz bleibt Teil des Nachrichtentextes. Kein Lerntipp, kein Fazit über Deutschlernen, keine Motivation.
    - Füge keine neuen Ergebnisse, Prognosen, beendeten Warnungen, gelungenen Abstimmungen, Verletzungen, Ursachen oder Folgen hinzu, wenn sie nicht in der Quelle stehen.
    - Schreibe keine spekulativen Zukunftssätze wie "Er wird nun versuchen ...", außer die Quelle sagt das ausdrücklich.
    - Für A1/A2: kurze, konkrete Sätze; einfache Gegenwartsformen; wenig Nebensätze; abstrakte politische oder technische Begriffe nur, wenn sie einfach erklärt werden.
    - Für A1/A2: prüfe besonders Artikel, Adjektivendungen, Singular/Plural und Verbformen.
    - Für B1/B2: nutze mehr Konnektoren und etwas Fachwortschatz, aber erkläre Zusammenhänge weiterhin klar und lesbar.
    - Für C1/C2: Nuancen, Argumentation und komplexere Satzstrukturen sind erlaubt, solange der Text klar bleibt.
    - Alle fünf Absätze behandeln die Nachricht selbst.
    """,
    output_key="base_text",
)

memo_agent = LlmAgent(
    name="VokabelnAgent",
    model=build_model(),
    description="Wählt schwierige Wörter aus und liefert Übersetzungen.",
    instruction=LESSON_STYLE_GUIDE
    + """Du bist eine Wortschatzlehrkraft und wählst nützliche Wörter aus dem Lesetext aus.

    Eingabe:
    - Text: {base_text}
    - Zielniveau nach GER: {level}

    Aufgabe:
    - Wähle genau 5 nützliche Wörter oder kurze Ausdrücke, die im Text vorkommen.
    - Wähle Wörter, die für Lernende auf Niveau {level} anspruchsvoll, aber realistisch sind.
    - Bevorzuge thematisch wichtige Wörter statt allgemeiner Wörter.
    - Bei Nomen gib Artikel und Plural in dieser Form an: "der Konflikt, -e" oder "die Reform, -en". Wenn der Plural unüblich ist, schreibe "kein Plural".
    - Wenn ein Nomen im Text im Plural steht, gib die Wörterbuchform im Singular an, z. B. "die Person, -en", nicht "die Personen, kein Plural".
    - Falsch: "3. 3. **die Einträge, -e**". Richtig: "3. **der Eintrag, -e**".
    - Falsch: "4. **die Informationen, -n**". Richtig: "4. **die Information, -en**".
    - Die deutsche Definition muss einfacher sein als der Lesetext.
    - Definitionen müssen natürliches Deutsch sein, keine wörtlichen Übersetzungen aus dem Englischen.
    - Beispielsätze müssen natürlich klingen. Schreibe nicht "Das Alter ist groß"; schreibe einen einfachen, idiomatischen Satz.
    - Erkläre die Bedeutung im Kontext des Textes. Wenn ein Wort im Text bildlich verwendet wird, erkläre nicht nur die wörtliche Bedeutung.
    - Verwende für jeden Eintrag genau dieses einzeilige Markdown-Format, inklusive fettgedrucktem Wort und den wörtlichen Labels:
      1. **der Konflikt, -e** - Definition: ein Streit zwischen Gruppen - Englisch: conflict - Beispiel: Der Konflikt dauert lange.
    - Verdopple die Nummer nicht. Richtig: "1. **Wort**"; falsch: "1. 1. **Wort**".
    - Ein Eintrag ist ungültig, wenn eines dieser drei Labels fehlt: "Definition:", "Englisch:" und "Beispiel:".
    - Nummeriere die Einträge von 1 bis 5. Füge keine zusätzlichen Notizen hinzu.
    """,
    output_key="vokabeln",
)

understand_agent = LlmAgent(
    name="VerstandenAgent",
    model=build_model(),
    description="Erstellt Verständnisfragen.",
    instruction=LESSON_STYLE_GUIDE
    + """Du bist eine Lehrkraft für Leseverstehen und formulierst klare Verständnisfragen.

    Eingabe:
    - Text: {base_text}
    - Zielniveau nach GER: {level}

    Aufgabe:
    - Erstelle genau 3 Verständnisfragen auf Deutsch.
    - Nummeriere sie von 1 bis 3.
    - Frage 1 fragt nach der Hauptaussage.
    - Frage 2 fragt nach einem konkreten Detail aus dem Text.
    - Frage 3 fragt nach einer einfachen Schlussfolgerung oder Meinung, die klar auf dem Text basiert.
    - Jede Frage muss mit Informationen aus dem Text beantwortbar sein.
    - Frage 1 fragt sachlich nach der Hauptaussage, nicht nach einer persönlichen Beschreibung.
    - Jede Frage besteht aus genau einem Satz.
    - Vermeide mehrteilige Fragen, besonders für A1, A2, B1 und B2.
    - Formuliere die Fragen natürlich und niveaugerecht, ohne formelle Anrede.
    - Prüfe jede Frage auf sauberes Deutsch. Keine fremdsprachlichen Wörter oder Tippfehler wie "la Umgebung".
    """,
    output_key="understanding_questions",
)

grammar_agent = LlmAgent(
    name="GrammatikAgent",
    model=build_model(),
    description="Erklärt eine Grammatikregel aus dem Text.",
    instruction=LESSON_STYLE_GUIDE
    + """Du bist eine Deutschlehrkraft für Grammatik und erklärst ein nützliches Muster aus dem Lesetext.

    Eingabe:
    - Text: {base_text}
    - Zielniveau nach GER: {level}

    Aufgabe:
    - Wähle genau ein Grammatikthema, das im Text vorkommt und zum Niveau {level} passt.
    - Wähle ein niveaugerechtes Grammatikthema:
      - A1/A2: Wortstellung, Modalverben, Präsens, Kasus, Artikel, Negation mit "nicht"/"kein".
      - B1/B2: Nebensätze, Passiv, Relativsätze, Konnektoren, Verbposition.
      - C1/C2: Nominalisierung, Konjunktiv I/II, Partizipialkonstruktionen, komplexer Satzbau.
    - Zitiere einen vollständigen Beispielsatz wortgetreu aus dem Text. Das Zitat muss exakt so im Text stehen.
    - Das Beispiel muss ein vollständiger Satz aus dem Lesetext sein, nicht nur ein Nebensatz oder Satzteil.
    - Erkläre die Regel in klarem Deutsch und einfacher als der Lesetext.
    - Verwende präzise Begriffe: "konjugiertes Verb", "Infinitiv", "Subjekt", "Objekt", "Artikel", "Kasus". Schreibe nicht "Verbstamm", wenn du eine Verbform meinst.
    - Erkläre Regeln als Fakten, nicht als Befehle. Kein "du musst", kein "du sollst".
    - Die Erklärung ist kurz: höchstens 3 bis 5 Sätze.
    - Das Mini-Muster muss ein grammatisch korrekter deutscher Beispielsatz sein und darf nicht mit "Sie" beginnen.
    - Verwende keine Auslassungspunkte ("...") im fertigen Grammatikabschnitt.
    - Das Mini-Muster muss idiomatisch sein. Falsch: "das Wetter regnet"; richtig: "es regnet" oder "das Wetter ist schlecht".
    - Verwende genau dieses Markdown-Format:
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
    description="Erstellt eine Schreibaufgabe.",
    instruction=LESSON_STYLE_GUIDE
    + """Du bist eine Schreiblehrkraft und formulierst eine klare Aufgabe für eine selbstlernende Person.

    Eingabe:
    - Text: {base_text}
    - Zielniveau nach GER: {level}

    Aufgabe:
    - Erstelle genau eine Schreibaufgabe auf Deutsch, passend zum Niveau {level}.
    - Sprich die lernende Person ausschließlich mit "du" an.
    - Verwende nicht "Sie", "Ihnen", "Ihre", "Schreiben Sie" oder "Wählen Sie".
    - Erwähne diese Anrede-Regel nicht in der sichtbaren Aufgabe. Wende sie still an.
    - Die Aufgabe bleibt eng mit dem Artikelthema verbunden.
    - Formuliere eine einzige klare Schreibaufgabe, keine Auswahl aus mehreren Optionen.
    - Die Aufgabe muss mit dem Lesetext und einer persönlichen Meinung lösbar sein; kein externes Wissen verlangen.
    - Nenne eine Zieltextlänge: für A1/A2 60-80 Wörter; ab B1 etwa 100 Wörter.
    - Gib 3 oder 4 leitende Stichpunkte.
    - Schreibe nicht "du musst" oder "du sollst". Nutze direkte Aufgabenverben wie "Schreibe ..." oder "Beschreibe ...".
    - Die Aufgabe muss grammatisch korrekt als direkte Aufforderung formuliert sein. Falsch: "Schreibe ... und du äußerst ...". Richtig: "Schreibe ... und äußere ...".
    - Fülle die Vorlage vollständig aus. Lasse keine Platzhalter wie "..." oder "Thema" stehen.
    - Verwende genau diese Markdown-Form:
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
# Aggregieren + speichern (state-gebundener FunctionNode)
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
    """Fasst die generierten Übungen als Markdown zusammen und speichert sie.

    Parameter werden anhand ihrer Namen automatisch aus dem Workflow-State
    gebunden, also aus den ``output_key``-Werten der vorgelagerten Agenten und
    aus ``fetch_news``.

    Returns:
        Eine Bestätigungsmeldung gefolgt von der vollständigen Lektion.
    """
    # Link zum Originalartikel direkt unter dem Lektionstitel, falls vorhanden.
    source = f"\n**Quelle:** [{news_title or 'Originalartikel'}]({news_url})\n" if news_url else ""

    # Artikel-Seed + Niveau einbetten, damit dieselbe Nachricht später auf
    # einem anderen Niveau neu generiert werden kann. Der Kommentar ist im
    # gerenderten Markdown unsichtbar.
    meta = json.dumps(
        {
            "news_title": news_title,
            "news_url": news_url,
            "recent_news": recent_news,
            "level": level,
        },
        ensure_ascii=False,
    )

    lesson = f"""# Deutschübungen
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

    return f"Lektion gespeichert unter {filename}\n\n{lesson}"


# Ein JoinNode wartet, bis alle vier parallelen Agenten fertig sind, bevor die
# Lektion gespeichert wird. Ohne ihn würde `save_lesson` einmal pro parallelem
# Agenten ausgelöst und vier Teildateien speichern.
collect_exercises = JoinNode(name="CollectExercises")

# ---------------------------------------------------------------------------
# Workflow-Graph
# ---------------------------------------------------------------------------

root_agent = Workflow(
    name="GenerateExercisesWorkflow",
    description="Generiert vollständige Deutschübungen auf Basis aktueller Nachrichten.",
    edges=[
        # Sequenzieller Start: echten Artikel laden, dann den Lesetext schreiben.
        (START, fetch_news, writer_agent),
        # Auffächern: vier didaktische Agenten arbeiten parallel mit {base_text}.
        (
            writer_agent,
            (memo_agent, understand_agent, grammar_agent, writing_assignment_agent),
        ),
        # Zusammenführen: alle vier einsammeln, dann einmal aggregieren und speichern.
        (
            (memo_agent, understand_agent, grammar_agent, writing_assignment_agent),
            collect_exercises,
            save_lesson,
        ),
    ],
)
