"""Web UI for the German lesson generator.

A small FastAPI app to:
  - read previously generated lessons (rendered Markdown),
  - generate a new lesson from a real news article,
  - choose the CEFR level, topic and news feed.

Run it (from anywhere):

    uv run python scripts/ui.py

then open http://localhost:8000
"""

import glob
import json
import os
import re
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
# Lessons are written to / read from the project root, regardless of where the
# server was launched from.
os.chdir(PROJECT_ROOT)

import uvicorn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

from google.adk.runners import InMemoryRunner
from google.genai import types

import german_teacher.agent as ag

APP_NAME = "german_teacher"
UI_USER = "ui"
LESSON_GLOB = "german_lesson_*.md"
LESSON_NAME_RE = re.compile(r"^german_lesson_[0-9_]+\.md$")
QUELLE_RE = re.compile(r"\*\*Quelle:\*\*\s*\[(.+?)\]")
META_RE = re.compile(r"<!--\s*lesson-meta:\s*(\{.*?\})\s*-->", re.DOTALL)

FEEDS = [
    {"label": "Alle (Standard)", "url": "https://www.tagesschau.de/index~rss2.xml"},
    {"label": "Inland", "url": "https://www.tagesschau.de/inland/index~rss2.xml"},
    {"label": "Ausland", "url": "https://www.tagesschau.de/ausland/index~rss2.xml"},
    {"label": "Wirtschaft", "url": "https://www.tagesschau.de/wirtschaft/index~rss2.xml"},
    {"label": "Wissen", "url": "https://www.tagesschau.de/wissen/index~rss2.xml"},
    {"label": "Investigativ", "url": "https://www.tagesschau.de/investigativ/index~rss2.xml"},
]
LEVELS = ["A1", "A2", "B1", "B1/B2", "B2", "C1", "C2"]

app = FastAPI(title="German Lesson Studio")


class GenerateRequest(BaseModel):
    topic: str = ""
    level: str = ""
    feed: str = ""
    # When set, the workflow reuses this exact article instead of fetching a new
    # one — used to regenerate the same news at a different level.
    recent_news: str = ""
    news_title: str = ""
    news_url: str = ""


def _lesson_label(content: str) -> str:
    """Derives a friendly label from a lesson's source article, if present."""
    match = QUELLE_RE.search(content)
    return match.group(1).strip() if match else ""


def _lesson_level(content: str) -> str:
    """Reads the embedded CEFR level from a lesson's metadata, if present."""
    match = META_RE.search(content)
    if not match:
        return ""
    try:
        return json.loads(match.group(1)).get("level", "")
    except json.JSONDecodeError:
        return ""


def _list_lessons() -> list[dict]:
    lessons = []
    for path in glob.glob(LESSON_GLOB):
        name = os.path.basename(path)
        try:
            with open(path, encoding="utf-8") as f:
                content = f.read()
        except OSError:
            content = ""
        lessons.append(
            {
                "name": name,
                "mtime": os.path.getmtime(path),
                "label": _lesson_label(content),
                "level": _lesson_level(content),
            }
        )
    lessons.sort(key=lambda item: item["mtime"], reverse=True)
    return lessons


def _build_prompt(topic: str) -> str:
    topic = (topic or "").strip()
    if topic:
        return f"Bitte erstelle eine komplette Deutsch-Lektion zum Thema: {topic}."
    return "Bitte erstelle eine komplette Deutsch-Lektion."


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return INDEX_HTML


@app.get("/api/config")
def config() -> dict:
    return {"feeds": FEEDS, "levels": LEVELS, "default_level": ag.DEFAULT_LEVEL}


@app.get("/api/lessons")
def lessons() -> list[dict]:
    return _list_lessons()


@app.get("/api/lessons/{name}")
def lesson(name: str):
    if not LESSON_NAME_RE.match(name) or not os.path.exists(name):
        return JSONResponse({"error": "Lesson not found."}, status_code=404)
    with open(name, encoding="utf-8") as f:
        return {"name": name, "content": f.read()}


@app.post("/api/generate")
async def generate(req: GenerateRequest):
    before = set(glob.glob(LESSON_GLOB))

    state: dict = {}
    if req.level:
        state["level"] = req.level
    if req.feed:
        state["news_rss_url"] = req.feed
    # Pin the article so the same news is reused (level switch).
    if req.recent_news:
        state["recent_news"] = req.recent_news
        state["news_title"] = req.news_title
        state["news_url"] = req.news_url

    runner = InMemoryRunner(agent=ag.root_agent, app_name=APP_NAME)
    session = await runner.session_service.create_session(
        app_name=APP_NAME, user_id=UI_USER, state=state
    )
    message = types.Content(role="user", parts=[types.Part(text=_build_prompt(req.topic))])

    try:
        async for _ in runner.run_async(
            user_id=UI_USER, session_id=session.id, new_message=message
        ):
            pass
    except Exception as exc:  # noqa: BLE001 - surface the error to the UI
        return JSONResponse({"error": f"Generation failed: {exc}"}, status_code=500)

    new_files = sorted(set(glob.glob(LESSON_GLOB)) - before)
    if not new_files:
        return JSONResponse({"error": "No lesson file was produced."}, status_code=500)

    name = new_files[-1]
    with open(name, encoding="utf-8") as f:
        return {"name": name, "content": f.read()}


INDEX_HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>German Lesson Studio</title>
<script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
<style>
  :root { --bg:#0f172a; --panel:#1e293b; --muted:#94a3b8; --line:#334155;
          --accent:#38bdf8; --accent2:#0ea5e9; --text:#e2e8f0; }
  * { box-sizing: border-box; }
  body { margin:0; font-family: system-ui, "Segoe UI", Roboto, sans-serif;
         background:var(--bg); color:var(--text); height:100vh; display:flex; flex-direction:column; }
  header { padding:14px 20px; border-bottom:1px solid var(--line); display:flex;
           align-items:center; gap:10px; }
  header h1 { font-size:18px; margin:0; font-weight:600; }
  header .tag { color:var(--muted); font-size:13px; }
  main { flex:1; display:grid; grid-template-columns: 320px 1fr; min-height:0; }
  aside { border-right:1px solid var(--line); display:flex; flex-direction:column; min-height:0; }
  .gen { padding:16px; border-bottom:1px solid var(--line); }
  .gen h2 { font-size:13px; text-transform:uppercase; letter-spacing:.05em;
            color:var(--muted); margin:0 0 10px; }
  label { display:block; font-size:12px; color:var(--muted); margin:10px 0 4px; }
  input, select, button { width:100%; padding:8px 10px; border-radius:8px;
            border:1px solid var(--line); background:#0b1220; color:var(--text); font-size:14px; }
  .row { display:flex; gap:8px; }
  .row > div { flex:1; }
  button.primary { margin-top:14px; background:linear-gradient(180deg,var(--accent),var(--accent2));
            border:none; color:#04222e; font-weight:600; cursor:pointer; }
  button.primary:disabled { opacity:.6; cursor:default; }
  #status { font-size:12px; color:var(--muted); margin-top:8px; min-height:16px; }
  .list { overflow:auto; flex:1; }
  .list h2 { font-size:13px; text-transform:uppercase; letter-spacing:.05em;
            color:var(--muted); margin:0; padding:14px 16px 6px; position:sticky; top:0; background:transparent; }
  .item { padding:10px 16px; border-bottom:1px solid #1f2c44; cursor:pointer; }
  .item:hover { background:#172033; }
  .item.active { background:#0b2b3a; border-left:3px solid var(--accent); }
  .item .t { font-size:14px; font-weight:500; }
  .item .s { font-size:12px; color:var(--muted); margin-top:2px; }
  section.view { overflow:auto; padding:0; display:flex; flex-direction:column; }
  .leveler { display:flex; align-items:center; gap:10px; flex-wrap:wrap;
             padding:12px 36px; background:#0b2b3a; border-bottom:1px solid var(--line);
             position:sticky; top:0; z-index:2; font-size:13px; }
  .leveler .ll-label { color:var(--text); }
  .leveler select { width:auto; }
  .leveler button { width:auto; padding:6px 14px; background:linear-gradient(180deg,var(--accent),var(--accent2));
             border:none; color:#04222e; font-weight:600; cursor:pointer; }
  .leveler button:disabled { opacity:.6; cursor:default; }
  .leveler .ll-status { color:var(--muted); }
  #view { padding:28px 36px; }
  .empty { color:var(--muted); margin-top:60px; text-align:center; }
  .md { max-width:820px; line-height:1.6; }
  .md h1 { font-size:26px; border-bottom:1px solid var(--line); padding-bottom:8px; }
  .md h2 { font-size:19px; margin-top:28px; color:var(--accent); }
  .md a { color:var(--accent); }
  .md code { background:#0b1220; padding:2px 5px; border-radius:4px; }
  .badge { display:inline-block; font-size:11px; background:#0b2b3a; color:var(--accent);
           border:1px solid var(--accent); border-radius:999px; padding:1px 8px; margin-left:8px; }
</style>
</head>
<body>
<header>
  <h1>🇩🇪 German Lesson Studio</h1>
  <span class="tag">read · generate · adjust level</span>
</header>
<main>
  <aside>
    <div class="gen">
      <h2>Generate a lesson</h2>
      <label for="topic">Topic (optional)</label>
      <input id="topic" placeholder="z.B. Sport, Wirtschaft, Klima…">
      <div class="row">
        <div>
          <label for="level">Level</label>
          <select id="level"></select>
        </div>
        <div>
          <label for="feed">News feed</label>
          <select id="feed"></select>
        </div>
      </div>
      <button class="primary" id="gen">Generate lesson</button>
      <div id="status"></div>
    </div>
    <div class="list">
      <h2>Lessons</h2>
      <div id="lessons"></div>
    </div>
  </aside>
  <section class="view">
    <div id="leveler" class="leveler" style="display:none"></div>
    <div id="view" class="md"><div class="empty">Select a lesson on the left, or generate a new one.</div></div>
  </section>
</main>
<script>
const $ = (id) => document.getElementById(id);
const META_RE = /<!--\s*lesson-meta:\s*(\{[\s\S]*?\})\s*-->/;
let current = null;
let cfg = null;

function safeJson(s) { try { return JSON.parse(s); } catch (e) { return null; } }

async function loadConfig() {
  cfg = await (await fetch('/api/config')).json();
  $('level').innerHTML = cfg.levels.map(l =>
     `<option ${l === cfg.default_level ? 'selected' : ''}>${l}</option>`).join('');
  $('feed').innerHTML = cfg.feeds.map(f =>
     `<option value="${f.url}">${f.label}</option>`).join('');
}

function fmtTime(mtime) {
  const d = new Date(mtime * 1000);
  return d.toLocaleString();
}

async function loadLessons() {
  const items = await (await fetch('/api/lessons')).json();
  const box = $('lessons');
  if (!items.length) { box.innerHTML = '<div class="item s">No lessons yet.</div>'; return; }
  box.innerHTML = items.map(it => `
    <div class="item ${it.name === current ? 'active' : ''}" data-name="${it.name}">
      <div class="t">${it.label || it.name}</div>
      <div class="s">${fmtTime(it.mtime)}${it.level ? ' · ' + it.level : ''}</div>
    </div>`).join('');
  box.querySelectorAll('.item').forEach(el =>
    el.addEventListener('click', () => openLesson(el.dataset.name)));
}

function renderLesson(content) {
  const m = content.match(META_RE);
  const meta = m ? safeJson(m[1]) : null;
  const body = content.replace(META_RE, '').trim();
  const v = $('view');
  v.innerHTML = marked.parse(body);
  v.querySelectorAll('a').forEach(a => { a.target = '_blank'; a.rel = 'noopener'; });
  renderLeveler(meta);
}

function renderLeveler(meta) {
  const bar = $('leveler');
  if (!meta || !meta.recent_news) { bar.style.display = 'none'; bar.innerHTML = ''; return; }
  const opts = (cfg ? cfg.levels : []).map(l =>
     `<option ${l === meta.level ? 'selected' : ''}>${l}</option>`).join('');
  bar.style.display = 'flex';
  bar.innerHTML = `
    <span class="ll-label">📚 Same news${meta.level ? ' (currently <b>' + meta.level + '</b>)' : ''} — switch to:</span>
    <select id="ll-level">${opts}</select>
    <button id="ll-go">Regenerate at level</button>
    <span id="ll-status" class="ll-status"></span>`;
  $('ll-go').addEventListener('click', () => switchLevel(meta));
}

async function switchLevel(meta) {
  const level = $('ll-level').value;
  const go = $('ll-go'); go.disabled = true;
  $('ll-status').textContent = 'Regenerating at ' + level + '… (~20–40s)';
  try {
    const res = await fetch('/api/generate', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ level, recent_news: meta.recent_news,
                             news_title: meta.news_title, news_url: meta.news_url })
    });
    const data = await res.json();
    if (!res.ok || data.error) { $('ll-status').textContent = '⚠ ' + (data.error || 'Failed'); go.disabled = false; return; }
    current = data.name;
    renderLesson(data.content);
    await loadLessons();
  } catch (e) { $('ll-status').textContent = '⚠ ' + e; go.disabled = false; }
}

async function openLesson(name) {
  current = name;
  const data = await (await fetch('/api/lessons/' + encodeURIComponent(name))).json();
  if (data.error) { renderLesson('**Error:** ' + data.error); return; }
  renderLesson(data.content);
  await loadLessons();
}

async function generate() {
  const btn = $('gen');
  btn.disabled = true;
  $('status').textContent = 'Generating… (this runs the full pipeline, ~20–40s)';
  try {
    const res = await fetch('/api/generate', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ topic: $('topic').value, level: $('level').value, feed: $('feed').value })
    });
    const data = await res.json();
    if (!res.ok || data.error) { $('status').textContent = '⚠ ' + (data.error || 'Failed'); return; }
    $('status').textContent = '✓ Created ' + data.name;
    current = data.name;
    renderLesson(data.content);
    await loadLessons();
  } catch (e) {
    $('status').textContent = '⚠ ' + e;
  } finally {
    btn.disabled = false;
  }
}

$('gen').addEventListener('click', generate);
loadConfig().then(loadLessons);
</script>
</body>
</html>
"""


if __name__ == "__main__":
    port = int(os.getenv("UI_PORT", "8000"))
    print(f"German Lesson Studio -> http://localhost:{port}")
    print(f"Provider: {'LM Studio' if ag.USING_LM_STUDIO else 'Gemini'}  |  default level: {ag.DEFAULT_LEVEL}")
    uvicorn.run(app, host="127.0.0.1", port=port)
