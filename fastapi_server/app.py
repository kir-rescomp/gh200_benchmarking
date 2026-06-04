"""
Minimal FastAPI proof-of-concept for SSH tunnel testing.
Serves a dashboard UI + /health endpoint.

Usage:
    uvicorn app:app --host 0.0.0.0 --port 8080
"""

import platform
import socket
import time
from datetime import datetime

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse

START_TIME = time.time()

app = FastAPI(title="GH200 Tunnel PoC")

@app.get("/health")
def health():
    return JSONResponse({
        "status": "ok",
        "hostname": socket.gethostname(),
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "uptime_seconds": round(time.time() - START_TIME, 1),
    })


@app.get("/info")
def info():
    return JSONResponse({
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cpu_count": platform.processor() or "unknown",
        "node": platform.node(),
    })


@app.get("/", response_class=HTMLResponse)
def index():
    return HTMLResponse(HTML)


HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>GH200 Tunnel PoC</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;600;700&family=Syne:wght@400;700;800&display=swap');

  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

  :root {
    --bg:       #0a0e17;
    --surface:  #111827;
    --border:   #1e2d45;
    --accent:   #00e5ff;
    --accent2:  #7c3aed;
    --green:    #10b981;
    --red:      #ef4444;
    --text:     #e2e8f0;
    --muted:    #64748b;
    --glow:     0 0 20px rgba(0,229,255,0.15);
  }

  body {
    background: var(--bg);
    color: var(--text);
    font-family: 'JetBrains Mono', monospace;
    min-height: 100vh;
    display: flex;
    flex-direction: column;
    align-items: center;
    padding: 2rem 1rem;
    background-image:
      radial-gradient(ellipse 80% 50% at 50% -10%, rgba(124,58,237,0.12) 0%, transparent 60%),
      radial-gradient(ellipse 60% 40% at 80% 80%, rgba(0,229,255,0.06) 0%, transparent 50%);
  }

  header {
    text-align: center;
    margin-bottom: 2.5rem;
    animation: fadeDown 0.6s ease both;
  }

  header h1 {
    font-family: 'JetBrains Mono', monospace;
    font-size: clamp(1rem, 2vw, 1.4rem);
    font-weight: 600;
    letter-spacing: 0.05em;
    background: linear-gradient(135deg, var(--accent) 0%, var(--accent2) 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: var(--text);
    background: none;
    background-clip: text;
  }

  header p {
    color: var(--muted);
    font-size: 0.78rem;
    margin-top: 0.4rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
  }

  .grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
    gap: 1.2rem;
    width: 100%;
    max-width: 860px;
    animation: fadeUp 0.6s ease 0.15s both;
  }

  .card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.4rem 1.6rem;
    position: relative;
    overflow: hidden;
    transition: border-color 0.2s, box-shadow 0.2s;
  }

  .card:hover {
    border-color: rgba(0,229,255,0.3);
    box-shadow: var(--glow);
  }

  .card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, var(--accent2), var(--accent));
    opacity: 0;
    transition: opacity 0.2s;
  }
  .card:hover::before { opacity: 1; }

  .card-label {
    font-size: 0.65rem;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: var(--muted);
    margin-bottom: 0.8rem;
  }

  .status-row {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    margin-bottom: 1rem;
  }

  .dot {
    width: 10px; height: 10px;
    border-radius: 50%;
    background: var(--muted);
    flex-shrink: 0;
    transition: background 0.3s, box-shadow 0.3s;
  }
  .dot.ok   { background: var(--green); box-shadow: 0 0 8px var(--green); }
  .dot.err  { background: var(--red);   box-shadow: 0 0 8px var(--red); }
  .dot.ping { animation: pulse 1s ease infinite; }

  @keyframes pulse {
    0%,100% { opacity: 1; } 50% { opacity: 0.3; }
  }

  #status-text {
    font-size: 1rem;
    font-weight: 600;
    color: var(--text);
  }

  .kv { display: flex; flex-direction: column; gap: 0.5rem; }
  .kv-row {
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    gap: 1rem;
    font-size: 0.78rem;
    border-bottom: 1px solid var(--border);
    padding-bottom: 0.4rem;
  }
  .kv-row:last-child { border-bottom: none; padding-bottom: 0; }
  .kv-key   { color: var(--muted); flex-shrink: 0; }
  .kv-value { color: var(--accent); text-align: right; word-break: break-all; }

  .log-card { grid-column: 1 / -1; }
  .log-box {
    background: #070b12;
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 0.9rem 1rem;
    height: 140px;
    overflow-y: auto;
    font-size: 0.72rem;
    line-height: 1.7;
    color: var(--muted);
  }
  .log-box::-webkit-scrollbar { width: 4px; }
  .log-box::-webkit-scrollbar-thumb { background: var(--border); border-radius: 2px; }
  .log-entry { display: block; }
  .log-entry .ts { color: var(--accent2); }
  .log-entry .ok-msg { color: var(--green); }
  .log-entry .err-msg { color: var(--red); }

  .btn-row {
    display: flex; gap: 0.8rem; margin-top: 1.2rem; flex-wrap: wrap;
    animation: fadeUp 0.6s ease 0.3s both;
  }

  button {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.05em;
    padding: 0.55rem 1.2rem;
    border-radius: 6px;
    border: 1px solid var(--border);
    background: var(--surface);
    color: var(--text);
    cursor: pointer;
    transition: all 0.15s;
    text-transform: uppercase;
  }
  button:hover { border-color: var(--accent); color: var(--accent); box-shadow: var(--glow); }
  button.primary {
    background: linear-gradient(135deg, var(--accent2), var(--accent));
    border: none; color: #000; font-weight: 700;
  }
  button.primary:hover { opacity: 0.85; color: #000; box-shadow: 0 0 20px rgba(0,229,255,0.3); }

  #uptime-val { font-variant-numeric: tabular-nums; }

  @keyframes fadeDown {
    from { opacity: 0; transform: translateY(-16px); }
    to   { opacity: 1; transform: translateY(0); }
  }
  @keyframes fadeUp {
    from { opacity: 0; transform: translateY(16px); }
    to   { opacity: 1; transform: translateY(0); }
  }
</style>
</head>
<body>

<header>
  <h1>GH200 · SSH Tunnel PoC</h1>
  <p>SSH port-forward proof of concept · BMRC</p>
</header>

<div class="grid">

  <!-- Health card -->
  <div class="card">
    <div class="card-label">Health check · /health</div>
    <div class="status-row">
      <div class="dot ping" id="dot"></div>
      <span id="status-text">Checking…</span>
    </div>
    <div class="kv" id="health-kv">
      <div class="kv-row"><span class="kv-key">hostname</span><span class="kv-value" id="h-host">—</span></div>
      <div class="kv-row"><span class="kv-key">uptime</span><span class="kv-value" id="uptime-val">—</span></div>
      <div class="kv-row"><span class="kv-key">server time</span><span class="kv-value" id="h-time">—</span></div>
    </div>
  </div>

  <!-- Info card -->
  <div class="card">
    <div class="card-label">Node info · /info</div>
    <div class="kv" id="info-kv">
      <div class="kv-row"><span class="kv-key">platform</span><span class="kv-value" id="i-platform">—</span></div>
      <div class="kv-row"><span class="kv-key">python</span><span class="kv-value" id="i-python">—</span></div>
      <div class="kv-row"><span class="kv-key">node</span><span class="kv-value" id="i-node">—</span></div>
    </div>
  </div>

  <!-- Log card -->
  <div class="card log-card">
    <div class="card-label">Poll log</div>
    <div class="log-box" id="log"></div>
  </div>

</div>

<div class="btn-row">
  <button class="primary" onclick="pollNow()">↻ Poll now</button>
  <button onclick="toggleAuto()">⏸ Pause auto-poll</button>
  <button onclick="clearLog()">✕ Clear log</button>
</div>

<script>
  let autoInterval = null;
  let autoPaused = false;
  let pollCount = 0;

  function ts() {
    return new Date().toISOString().split('T')[1].slice(0,8);
  }

  function log(msg, cls='') {
    const box = document.getElementById('log');
    const el = document.createElement('span');
    el.className = 'log-entry';
    el.innerHTML = `<span class="ts">[${ts()}]</span> <span class="${cls}">${msg}</span>\n`;
    box.appendChild(el);
    box.scrollTop = box.scrollHeight;
  }

  function fmtUptime(s) {
    const h = Math.floor(s / 3600);
    const m = Math.floor((s % 3600) / 60);
    const sec = Math.floor(s % 60);
    return [h && `${h}h`, m && `${m}m`, `${sec}s`].filter(Boolean).join(' ');
  }

  async function pollHealth() {
    const dot = document.getElementById('dot');
    dot.className = 'dot ping';
    try {
      const r = await fetch('/health');
      const d = await r.json();
      dot.className = 'dot ok';
      document.getElementById('status-text').textContent = 'OK';
      document.getElementById('h-host').textContent = d.hostname;
      document.getElementById('uptime-val').textContent = fmtUptime(d.uptime_seconds);
      document.getElementById('h-time').textContent = d.timestamp.replace('T',' ');
      log(`/health → ok · host=${d.hostname} uptime=${fmtUptime(d.uptime_seconds)}`, 'ok-msg');
    } catch(e) {
      dot.className = 'dot err';
      document.getElementById('status-text').textContent = 'Unreachable';
      log(`/health → ${e.message}`, 'err-msg');
    }
  }

  async function pollInfo() {
    try {
      const r = await fetch('/info');
      const d = await r.json();
      document.getElementById('i-platform').textContent = d.platform.slice(0,40);
      document.getElementById('i-python').textContent = d.python;
      document.getElementById('i-node').textContent = d.node;
    } catch(_) {}
  }

  async function pollNow() {
    pollCount++;
    await Promise.all([pollHealth(), pollCount === 1 ? pollInfo() : Promise.resolve()]);
  }

  function toggleAuto() {
    autoPaused = !autoPaused;
    const btn = document.querySelector('button:nth-child(2)');
    if (autoPaused) {
      clearInterval(autoInterval);
      btn.textContent = '▶ Resume auto-poll';
      log('Auto-poll paused');
    } else {
      autoInterval = setInterval(pollNow, 5000);
      btn.textContent = '⏸ Pause auto-poll';
      log('Auto-poll resumed');
    }
  }

  function clearLog() {
    document.getElementById('log').innerHTML = '';
  }

  // Boot
  pollNow();
  pollInfo();
  autoInterval = setInterval(pollNow, 5000);
  log('Dashboard started · polling every 5s');
</script>
</body>
</html>
"""
