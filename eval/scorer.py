#!/usr/bin/env python3
"""Moltbook RAG eval scoring GUI."""

import json
from pathlib import Path
from flask import Flask, render_template_string, jsonify, request

EVAL_DIR = Path(__file__).parent
QUESTIONS_FILE = EVAL_DIR / "questions.json"
RESULTS_FILE = EVAL_DIR / "results.json"
SCORES_FILE = EVAL_DIR / "scores.json"

app = Flask(__name__)


def load_json(path):
    if path.exists():
        return json.loads(path.read_text())
    return {} if path == SCORES_FILE else []


@app.route("/")
def index():
    return render_template_string(HTML)


@app.route("/api/data")
def api_data():
    questions = load_json(QUESTIONS_FILE)
    results = load_json(RESULTS_FILE)
    scores = load_json(SCORES_FILE)
    q_lookup = {q["id"]: q for q in questions}
    merged = []
    for r in results:
        q = q_lookup.get(r["id"], {})
        merged.append({
            "id": r["id"],
            "category": r.get("category", q.get("category", "")),
            "query": r.get("query", q.get("query", "")),
            "notes": q.get("notes", ""),
            "expected_authors": q.get("expected_authors", []),
            "expected_submolts": q.get("expected_submolts", []),
            "expected_title_keywords": q.get("expected_title_keywords", []),
            "modes": r.get("modes", {}),
        })
    return jsonify({"items": merged, "scores": scores})


@app.route("/api/score", methods=["POST"])
def api_score():
    data = request.json
    scores = load_json(SCORES_FILE)
    scores[data["id"]] = {"score": data["score"], "notes": data.get("notes")}
    SCORES_FILE.write_text(json.dumps(scores, indent=2) + "\n")
    return jsonify({"ok": True, "total_scored": len(scores)})


HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Moltbook RAG Eval</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=JetBrains+Mono:wght@300;400;500;600&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap" rel="stylesheet">
<style>
  :root {
    --bg: #0e0f13;
    --bg-card: #16171d;
    --bg-elevated: #1c1d25;
    --bg-mode: #12131a;
    --border: #2a2b35;
    --border-active: #3d3e4a;
    --text: #c8c9d0;
    --text-dim: #6b6c78;
    --text-bright: #eaebf0;
    --accent: #5b8def;
    --score-0: #e05252;
    --score-0-bg: rgba(224,82,82,0.08);
    --score-1: #e0943a;
    --score-1-bg: rgba(224,148,58,0.08);
    --score-2: #c9b84a;
    --score-2-bg: rgba(201,184,74,0.08);
    --score-3: #4aba7a;
    --score-3-bg: rgba(74,186,122,0.08);
    --cat-factual: #5b8def;
    --cat-synthesis: #a878e8;
    --cat-temporal: #e0943a;
    --cat-negative: #e05252;
    --cat-author: #4aba7a;
    --cat-mode_comparison: #c9b84a;
    --cat-community: #e078a8;
    --cat-submolt: #58c4dc;
    --cat-edge_case: #6b6c78;
  }

  * { margin: 0; padding: 0; box-sizing: border-box; }

  body {
    background: var(--bg);
    color: var(--text);
    font-family: 'IBM Plex Sans', sans-serif;
    font-weight: 400;
    min-height: 100vh;
    overflow-x: hidden;
  }

  /* ── Top bar ── */
  .topbar {
    position: fixed; top: 0; left: 0; right: 0; z-index: 100;
    display: flex; align-items: center; justify-content: space-between;
    padding: 0 32px; height: 56px;
    background: rgba(14,15,19,0.85);
    backdrop-filter: blur(20px);
    border-bottom: 1px solid var(--border);
  }
  .topbar-title {
    font-family: 'DM Serif Display', serif;
    font-size: 18px; color: var(--text-bright);
    letter-spacing: 0.02em;
  }
  .topbar-title span { color: var(--text-dim); font-weight: 300; }
  .topbar-right { display: flex; align-items: center; gap: 20px; }
  .progress-wrap {
    display: flex; align-items: center; gap: 10px;
    font-family: 'JetBrains Mono', monospace; font-size: 12px;
    color: var(--text-dim);
  }
  .progress-bar {
    width: 140px; height: 3px; background: var(--border); border-radius: 2px;
    overflow: hidden;
  }
  .progress-fill {
    height: 100%; background: var(--accent); border-radius: 2px;
    transition: width 0.4s cubic-bezier(0.4,0,0.2,1);
  }
  .btn-summary {
    font-family: 'JetBrains Mono', monospace; font-size: 11px;
    padding: 6px 14px; border: 1px solid var(--border); border-radius: 4px;
    background: transparent; color: var(--text-dim); cursor: pointer;
    transition: all 0.2s;
    text-transform: uppercase; letter-spacing: 0.08em;
  }
  .btn-summary:hover { border-color: var(--accent); color: var(--accent); }

  /* ── Main layout ── */
  .main { padding: 80px 32px 32px; max-width: 1100px; margin: 0 auto; }

  /* ── Navigation strip ── */
  .nav-strip {
    display: flex; gap: 4px; flex-wrap: wrap; margin-bottom: 24px;
    padding: 12px 0;
  }
  .nav-dot {
    width: 28px; height: 28px; border-radius: 4px;
    display: flex; align-items: center; justify-content: center;
    font-family: 'JetBrains Mono', monospace; font-size: 9px;
    cursor: pointer; transition: all 0.15s;
    border: 1px solid transparent;
    color: var(--text-dim); background: var(--bg-card);
  }
  .nav-dot:hover { border-color: var(--border-active); color: var(--text); }
  .nav-dot.active { border-color: var(--accent); color: var(--accent); background: rgba(91,141,239,0.08); }
  .nav-dot.scored { background: var(--bg-elevated); color: var(--text); }
  .nav-dot.scored::after {
    content: ''; position: absolute; bottom: 2px; left: 50%; transform: translateX(-50%);
    width: 4px; height: 4px; border-radius: 50%;
  }
  .nav-dot { position: relative; }
  .nav-dot.scored-0 { border-bottom: 2px solid var(--score-0); }
  .nav-dot.scored-1 { border-bottom: 2px solid var(--score-1); }
  .nav-dot.scored-2 { border-bottom: 2px solid var(--score-2); }
  .nav-dot.scored-3 { border-bottom: 2px solid var(--score-3); }

  /* ── Card ── */
  .card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 8px;
    overflow: hidden;
    animation: fadeUp 0.3s cubic-bezier(0.4,0,0.2,1);
  }
  @keyframes fadeUp {
    from { opacity: 0; transform: translateY(8px); }
    to { opacity: 1; transform: translateY(0); }
  }

  .card-header {
    padding: 24px 28px 20px;
    border-bottom: 1px solid var(--border);
  }
  .card-meta {
    display: flex; align-items: center; gap: 10px; margin-bottom: 12px;
  }
  .card-id {
    font-family: 'JetBrains Mono', monospace; font-size: 13px;
    color: var(--text-dim); font-weight: 500;
  }
  .cat-badge {
    font-family: 'JetBrains Mono', monospace; font-size: 10px;
    padding: 3px 10px; border-radius: 3px;
    text-transform: uppercase; letter-spacing: 0.06em;
    font-weight: 500;
  }
  .cat-factual { color: var(--cat-factual); background: rgba(91,141,239,0.1); }
  .cat-synthesis { color: var(--cat-synthesis); background: rgba(168,120,232,0.1); }
  .cat-temporal { color: var(--cat-temporal); background: rgba(224,148,58,0.1); }
  .cat-negative { color: var(--cat-negative); background: rgba(224,82,82,0.1); }
  .cat-author { color: var(--cat-author); background: rgba(74,186,122,0.1); }
  .cat-mode_comparison { color: var(--cat-mode_comparison); background: rgba(201,184,74,0.1); }
  .cat-community { color: var(--cat-community); background: rgba(224,120,168,0.1); }
  .cat-submolt { color: var(--cat-submolt); background: rgba(88,196,220,0.1); }
  .cat-edge_case { color: var(--cat-edge_case); background: rgba(107,108,120,0.15); }

  .card-query {
    font-family: 'DM Serif Display', serif;
    font-size: 22px; color: var(--text-bright);
    line-height: 1.35; margin-bottom: 14px;
  }
  .card-notes {
    font-size: 13px; color: var(--text-dim); line-height: 1.55;
    font-style: italic;
  }

  .expectations {
    display: flex; gap: 16px; flex-wrap: wrap;
    padding: 14px 28px; border-bottom: 1px solid var(--border);
    background: rgba(0,0,0,0.15);
  }
  .expect-group { display: flex; align-items: center; gap: 6px; }
  .expect-label {
    font-family: 'JetBrains Mono', monospace; font-size: 10px;
    color: var(--text-dim); text-transform: uppercase; letter-spacing: 0.05em;
  }
  .expect-tag {
    font-family: 'JetBrains Mono', monospace; font-size: 11px;
    padding: 2px 8px; border-radius: 3px;
    background: var(--bg-elevated); color: var(--text);
    border: 1px solid var(--border);
  }

  /* ── Mode sections ── */
  .modes-container { padding: 0; }
  .mode-section { border-bottom: 1px solid var(--border); }
  .mode-section:last-child { border-bottom: none; }
  .mode-header {
    display: flex; align-items: center; justify-content: space-between;
    padding: 12px 28px; cursor: pointer;
    transition: background 0.15s;
  }
  .mode-header:hover { background: rgba(255,255,255,0.02); }
  .mode-name {
    font-family: 'JetBrains Mono', monospace; font-size: 12px;
    font-weight: 500; color: var(--text);
    display: flex; align-items: center; gap: 8px;
  }
  .mode-name .arrow {
    display: inline-block; transition: transform 0.2s;
    color: var(--text-dim); font-size: 10px;
  }
  .mode-name .arrow.open { transform: rotate(90deg); }
  .mode-time {
    font-family: 'JetBrains Mono', monospace; font-size: 11px;
    color: var(--text-dim);
  }
  .mode-body {
    max-height: 0; overflow: hidden;
    transition: max-height 0.3s cubic-bezier(0.4,0,0.2,1);
  }
  .mode-body.open { max-height: 2000px; }
  .mode-output {
    padding: 16px 28px 20px;
    background: var(--bg-mode);
    font-family: 'JetBrains Mono', monospace; font-size: 12px;
    line-height: 1.7; color: var(--text);
    white-space: pre-wrap; word-break: break-word;
    max-height: 400px; overflow-y: auto;
  }
  .mode-output::-webkit-scrollbar { width: 6px; }
  .mode-output::-webkit-scrollbar-track { background: transparent; }
  .mode-output::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
  .mode-error {
    color: var(--score-0); font-style: italic;
  }

  /* ── Highlight search result lines ── */
  .result-rank { color: var(--accent); font-weight: 600; }
  .result-author { color: var(--cat-author); }
  .result-submolt { color: var(--cat-submolt); }
  .result-score-high { color: var(--score-3); }

  /* ── Score panel ── */
  .score-panel {
    padding: 24px 28px;
    background: var(--bg-elevated);
    display: flex; align-items: center; gap: 20px; flex-wrap: wrap;
  }
  .score-buttons { display: flex; gap: 8px; }
  .score-btn {
    width: 64px; height: 44px; border-radius: 6px;
    border: 2px solid transparent;
    font-family: 'JetBrains Mono', monospace; font-size: 13px; font-weight: 600;
    cursor: pointer; transition: all 0.15s;
    display: flex; flex-direction: column; align-items: center; justify-content: center;
    gap: 2px;
  }
  .score-btn .label {
    font-size: 8px; font-weight: 400; text-transform: uppercase;
    letter-spacing: 0.05em; opacity: 0.7;
  }
  .score-btn-0 { background: var(--score-0-bg); color: var(--score-0); border-color: rgba(224,82,82,0.2); }
  .score-btn-0:hover, .score-btn-0.active { border-color: var(--score-0); background: rgba(224,82,82,0.15); }
  .score-btn-1 { background: var(--score-1-bg); color: var(--score-1); border-color: rgba(224,148,58,0.2); }
  .score-btn-1:hover, .score-btn-1.active { border-color: var(--score-1); background: rgba(224,148,58,0.15); }
  .score-btn-2 { background: var(--score-2-bg); color: var(--score-2); border-color: rgba(201,184,74,0.2); }
  .score-btn-2:hover, .score-btn-2.active { border-color: var(--score-2); background: rgba(201,184,74,0.15); }
  .score-btn-3 { background: var(--score-3-bg); color: var(--score-3); border-color: rgba(74,186,122,0.2); }
  .score-btn-3:hover, .score-btn-3.active { border-color: var(--score-3); background: rgba(74,186,122,0.15); }

  .notes-input {
    flex: 1; min-width: 200px; height: 44px;
    background: var(--bg-card); border: 1px solid var(--border);
    border-radius: 6px; padding: 0 14px;
    font-family: 'IBM Plex Sans', sans-serif; font-size: 13px;
    color: var(--text); outline: none;
    transition: border-color 0.2s;
  }
  .notes-input::placeholder { color: var(--text-dim); }
  .notes-input:focus { border-color: var(--accent); }

  .nav-buttons { display: flex; gap: 8px; margin-left: auto; }
  .nav-btn {
    padding: 10px 20px; border-radius: 6px;
    font-family: 'JetBrains Mono', monospace; font-size: 12px;
    border: 1px solid var(--border); background: transparent;
    color: var(--text-dim); cursor: pointer; transition: all 0.15s;
  }
  .nav-btn:hover { border-color: var(--text-dim); color: var(--text); }
  .nav-btn.primary {
    background: var(--accent); border-color: var(--accent);
    color: #fff;
  }
  .nav-btn.primary:hover { background: #4a7ce0; }
  .nav-btn:disabled { opacity: 0.3; cursor: default; }

  /* ── Keyboard hints ── */
  .kbd-hints {
    display: flex; gap: 16px; justify-content: center;
    padding: 16px; margin-top: 16px;
    font-family: 'JetBrains Mono', monospace; font-size: 11px;
    color: var(--text-dim);
  }
  .kbd-hints kbd {
    display: inline-block; padding: 2px 6px;
    background: var(--bg-elevated); border: 1px solid var(--border);
    border-radius: 3px; font-size: 10px; color: var(--text);
    margin: 0 2px;
  }

  /* ── Summary overlay ── */
  .overlay {
    position: fixed; inset: 0; z-index: 200;
    background: rgba(0,0,0,0.7); backdrop-filter: blur(8px);
    display: none; align-items: center; justify-content: center;
  }
  .overlay.visible { display: flex; }
  .summary-card {
    background: var(--bg-card); border: 1px solid var(--border);
    border-radius: 10px; padding: 36px 40px;
    min-width: 420px; max-width: 520px;
    animation: fadeUp 0.3s cubic-bezier(0.4,0,0.2,1);
  }
  .summary-title {
    font-family: 'DM Serif Display', serif;
    font-size: 24px; color: var(--text-bright);
    margin-bottom: 24px;
  }
  .summary-avg {
    font-family: 'JetBrains Mono', monospace;
    font-size: 48px; font-weight: 300;
    color: var(--text-bright); margin-bottom: 4px;
  }
  .summary-avg-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px; color: var(--text-dim);
    text-transform: uppercase; letter-spacing: 0.08em;
    margin-bottom: 28px;
  }
  .summary-bars { display: flex; flex-direction: column; gap: 10px; }
  .summary-row {
    display: flex; align-items: center; gap: 12px;
    font-family: 'JetBrains Mono', monospace; font-size: 13px;
  }
  .summary-row .s-label { width: 90px; color: var(--text-dim); }
  .summary-row .s-bar-wrap {
    flex: 1; height: 6px; background: var(--border); border-radius: 3px;
    overflow: hidden;
  }
  .summary-row .s-bar {
    height: 100%; border-radius: 3px;
    transition: width 0.5s cubic-bezier(0.4,0,0.2,1);
  }
  .summary-row .s-count { width: 30px; text-align: right; color: var(--text); }
  .summary-close {
    margin-top: 28px; width: 100%; padding: 10px;
    font-family: 'JetBrains Mono', monospace; font-size: 12px;
    border: 1px solid var(--border); border-radius: 6px;
    background: transparent; color: var(--text-dim); cursor: pointer;
    transition: all 0.2s;
  }
  .summary-close:hover { border-color: var(--text-dim); color: var(--text); }

  /* ── Category summary ── */
  .summary-cats { margin-top: 24px; border-top: 1px solid var(--border); padding-top: 20px; }
  .summary-cats h3 {
    font-family: 'JetBrains Mono', monospace; font-size: 11px;
    color: var(--text-dim); text-transform: uppercase; letter-spacing: 0.08em;
    margin-bottom: 12px;
  }
  .cat-row {
    display: flex; align-items: center; gap: 10px;
    font-family: 'JetBrains Mono', monospace; font-size: 12px;
    padding: 4px 0;
  }
  .cat-row .c-name { width: 120px; }
  .cat-row .c-avg { width: 50px; text-align: right; color: var(--text-bright); font-weight: 500; }
  .cat-row .c-count { color: var(--text-dim); }
</style>
</head>
<body>

<div class="topbar">
  <div class="topbar-title">Moltbook RAG Eval <span>/ scorer</span></div>
  <div class="topbar-right">
    <div class="progress-wrap">
      <span id="progress-text">0 / 37</span>
      <div class="progress-bar"><div class="progress-fill" id="progress-fill"></div></div>
    </div>
    <button class="btn-summary" onclick="toggleSummary()">Summary</button>
  </div>
</div>

<div class="main">
  <div class="nav-strip" id="nav-strip"></div>
  <div id="card-container"></div>
  <div class="kbd-hints">
    <span><kbd>0</kbd><kbd>1</kbd><kbd>2</kbd><kbd>3</kbd> score</span>
    <span><kbd>&larr;</kbd><kbd>&rarr;</kbd> navigate</span>
    <span><kbd>N</kbd> next unscored</span>
    <span><kbd>E</kbd> expand all</span>
  </div>
</div>

<div class="overlay" id="summary-overlay" onclick="if(event.target===this)toggleSummary()">
  <div class="summary-card" id="summary-card"></div>
</div>

<script>
let items = [], scores = {}, currentIdx = 0;

async function init() {
  const res = await fetch('/api/data');
  const data = await res.json();
  items = data.items;
  scores = data.scores;
  buildNav();
  render();
  updateProgress();
}

function buildNav() {
  const strip = document.getElementById('nav-strip');
  strip.innerHTML = items.map((item, i) => {
    const s = scores[item.id];
    let cls = 'nav-dot';
    if (i === currentIdx) cls += ' active';
    if (s) cls += ` scored scored-${s.score}`;
    return `<div class="${cls}" onclick="goTo(${i})" title="${item.id}: ${item.query.slice(0,40)}">${item.id}</div>`;
  }).join('');
}

function highlightOutput(raw) {
  if (!raw) return '<span class="mode-error">No output</span>';
  return raw
    .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
    .replace(/^(\s*\d+\.)/gm, '<span class="result-rank">$1</span>')
    .replace(/by\s+(\S+)/g, 'by <span class="result-author">$1</span>')
    .replace(/in\s+(\S+)\s*\|/g, 'in <span class="result-submolt">$1</span> |')
    .replace(/score:\s*(\d{2,})/g, 'score: <span class="result-score-high">$1</span>');
}

function render() {
  const item = items[currentIdx];
  const s = scores[item.id];
  const hasExpectations = item.expected_authors.length || item.expected_submolts.length || item.expected_title_keywords.length;

  const modeOrder = ['hybrid', 'keyword', 'semantic', 'hybrid_unfiltered'];
  const modeLabels = { hybrid: 'Hybrid (RRF)', keyword: 'Keyword Only', semantic: 'Semantic Only', hybrid_unfiltered: 'Hybrid (no filter)' };
  const availableModes = modeOrder.filter(m => item.modes[m]);

  let html = `<div class="card" key="${item.id}">
    <div class="card-header">
      <div class="card-meta">
        <span class="card-id">${item.id}</span>
        <span class="cat-badge cat-${item.category}">${item.category.replace('_', ' ')}</span>
      </div>
      <div class="card-query">${escHtml(item.query) || '<em>(empty query)</em>'}</div>
      ${item.notes ? `<div class="card-notes">${escHtml(item.notes)}</div>` : ''}
    </div>`;

  if (hasExpectations) {
    html += `<div class="expectations">`;
    if (item.expected_authors.length)
      html += `<div class="expect-group"><span class="expect-label">Authors</span>${item.expected_authors.map(a => `<span class="expect-tag">${escHtml(a)}</span>`).join('')}</div>`;
    if (item.expected_submolts.length)
      html += `<div class="expect-group"><span class="expect-label">Submolts</span>${item.expected_submolts.map(s => `<span class="expect-tag">${escHtml(s)}</span>`).join('')}</div>`;
    if (item.expected_title_keywords.length)
      html += `<div class="expect-group"><span class="expect-label">Keywords</span>${item.expected_title_keywords.map(k => `<span class="expect-tag">${escHtml(k)}</span>`).join('')}</div>`;
    html += `</div>`;
  }

  html += `<div class="modes-container">`;
  availableModes.forEach((mode, mi) => {
    const d = item.modes[mode];
    const isFirst = mi === 0;
    const timeStr = d.time_ms ? `${d.time_ms}ms` : '';
    const hasError = !!d.error;
    html += `
      <div class="mode-section">
        <div class="mode-header" onclick="toggleMode(this)">
          <span class="mode-name"><span class="arrow ${isFirst ? 'open' : ''}">\u25B6</span> ${modeLabels[mode] || mode}</span>
          <span class="mode-time">${timeStr}</span>
        </div>
        <div class="mode-body ${isFirst ? 'open' : ''}">
          <div class="mode-output">${hasError ? `<span class="mode-error">${escHtml(d.error)}</span>` : highlightOutput(d.raw_output)}</div>
        </div>
      </div>`;
  });
  html += `</div>`;

  html += `<div class="score-panel">
    <div class="score-buttons">
      ${[0,1,2,3].map(n => {
        const labels = ['Miss','Marginal','Relevant','Excellent'];
        const active = s && s.score === n ? ' active' : '';
        return `<button class="score-btn score-btn-${n}${active}" onclick="setScore(${n})">
          ${n}<span class="label">${labels[n]}</span>
        </button>`;
      }).join('')}
    </div>
    <input type="text" class="notes-input" id="notes-input"
      placeholder="Notes (optional)..."
      value="${s && s.notes ? escAttr(s.notes) : ''}"
      onkeydown="if(event.key==='Enter'){event.preventDefault();nextUnscored();}">
    <div class="nav-buttons">
      <button class="nav-btn" onclick="goTo(currentIdx-1)" ${currentIdx===0?'disabled':''}>&#8592; Prev</button>
      <button class="nav-btn primary" onclick="goTo(currentIdx+1)" ${currentIdx===items.length-1?'disabled':''}>Next &#8594;</button>
    </div>
  </div>`;

  html += `</div>`;
  document.getElementById('card-container').innerHTML = html;
}

function escHtml(s) { const d = document.createElement('div'); d.textContent = s; return d.innerHTML; }
function escAttr(s) { return s.replace(/"/g, '&quot;').replace(/'/g, '&#39;'); }

function toggleMode(el) {
  const arrow = el.querySelector('.arrow');
  const body = el.nextElementSibling;
  arrow.classList.toggle('open');
  body.classList.toggle('open');
}

function expandAllModes() {
  document.querySelectorAll('.mode-body').forEach(b => b.classList.add('open'));
  document.querySelectorAll('.arrow').forEach(a => a.classList.add('open'));
}

async function setScore(n) {
  const item = items[currentIdx];
  const notes = document.getElementById('notes-input')?.value || null;
  scores[item.id] = { score: n, notes };
  await fetch('/api/score', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ id: item.id, score: n, notes })
  });
  buildNav();
  updateProgress();
  render();
}

function goTo(idx) {
  if (idx < 0 || idx >= items.length) return;
  currentIdx = idx;
  buildNav();
  render();
  window.scrollTo({ top: 0, behavior: 'smooth' });
}

function nextUnscored() {
  for (let i = currentIdx + 1; i < items.length; i++) {
    if (!scores[items[i].id]) { goTo(i); return; }
  }
  for (let i = 0; i < currentIdx; i++) {
    if (!scores[items[i].id]) { goTo(i); return; }
  }
}

function updateProgress() {
  const scored = Object.keys(scores).length;
  const total = items.length;
  document.getElementById('progress-text').textContent = `${scored} / ${total}`;
  document.getElementById('progress-fill').style.width = `${(scored/total)*100}%`;
}

function toggleSummary() {
  const overlay = document.getElementById('summary-overlay');
  if (overlay.classList.contains('visible')) {
    overlay.classList.remove('visible');
    return;
  }
  const scored = Object.values(scores);
  if (!scored.length) { overlay.classList.add('visible'); document.getElementById('summary-card').innerHTML = `<div class="summary-title">No scores yet</div><button class="summary-close" onclick="toggleSummary()">Close</button>`; return; }

  const vals = scored.map(s => s.score);
  const avg = (vals.reduce((a,b) => a+b, 0) / vals.length).toFixed(2);
  const counts = [0,1,2,3].map(n => vals.filter(v => v === n).length);
  const max = Math.max(...counts, 1);

  const colors = ['var(--score-0)','var(--score-1)','var(--score-2)','var(--score-3)'];
  const labels = ['0 \u2014 Miss','1 \u2014 Marginal','2 \u2014 Relevant','3 \u2014 Excellent'];

  let catMap = {};
  items.forEach(item => {
    const s = scores[item.id];
    if (!s) return;
    const cat = item.category;
    if (!catMap[cat]) catMap[cat] = [];
    catMap[cat].push(s.score);
  });

  let catHtml = Object.entries(catMap).map(([cat, vals]) => {
    const a = (vals.reduce((a,b)=>a+b,0)/vals.length).toFixed(1);
    return `<div class="cat-row"><span class="c-name cat-badge cat-${cat}">${cat.replace('_',' ')}</span><span class="c-avg">${a}</span><span class="c-count">(${vals.length})</span></div>`;
  }).join('');

  document.getElementById('summary-card').innerHTML = `
    <div class="summary-title">Score Summary</div>
    <div class="summary-avg">${avg}</div>
    <div class="summary-avg-label">average score (${vals.length} / ${items.length} scored)</div>
    <div class="summary-bars">
      ${[3,2,1,0].map(n => `<div class="summary-row">
        <span class="s-label">${labels[n]}</span>
        <div class="s-bar-wrap"><div class="s-bar" style="width:${(counts[n]/max)*100}%;background:${colors[n]}"></div></div>
        <span class="s-count" style="color:${colors[n]}">${counts[n]}</span>
      </div>`).join('')}
    </div>
    <div class="summary-cats"><h3>By Category</h3>${catHtml}</div>
    <button class="summary-close" onclick="toggleSummary()">Close</button>`;
  overlay.classList.add('visible');
}

document.addEventListener('keydown', e => {
  if (e.target.tagName === 'INPUT') return;
  if (e.key >= '0' && e.key <= '3') { setScore(parseInt(e.key)); return; }
  if (e.key === 'ArrowLeft') { goTo(currentIdx - 1); return; }
  if (e.key === 'ArrowRight') { goTo(currentIdx + 1); return; }
  if (e.key === 'n' || e.key === 'N') { nextUnscored(); return; }
  if (e.key === 'e' || e.key === 'E') { expandAllModes(); return; }
  if (e.key === 'Escape') { document.getElementById('summary-overlay').classList.remove('visible'); }
});

init();
</script>
</body>
</html>"""


if __name__ == "__main__":
    print(f"  Eval dir: {EVAL_DIR}")
    print(f"  Questions: {QUESTIONS_FILE} ({'exists' if QUESTIONS_FILE.exists() else 'MISSING'})")
    print(f"  Results: {RESULTS_FILE} ({'exists' if RESULTS_FILE.exists() else 'MISSING'})")
    print(f"  Scores: {SCORES_FILE} ({'exists' if SCORES_FILE.exists() else 'new'})")
    print()
    app.run(host="127.0.0.1", port=5050, debug=False)
