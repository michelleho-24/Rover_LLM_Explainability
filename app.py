import io
import os
import threading
import webbrowser
import contextlib
import base64

from flask import Flask, request, jsonify

from RagFramework.agent import get_agent, ask

app = Flask(__name__)


@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.get_json(force=True)
    user_query = data.get("query", "").strip()
    if not user_query:
        return jsonify({"error": "empty query"}), 400

    plot_candidates = (
        "trajectory_plot.png", "trajectory_poi_add.png",
        "trajectory_poi_replace.png", "trajectory_custom_sites.png",
    )
    mtimes_before = {}
    for p in plot_candidates:
        if os.path.exists(p):
            mtimes_before[p] = os.path.getmtime(p)

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        answer = ask(user_query)

    plot_b64 = None
    for p in plot_candidates:
        if os.path.exists(p):
            new_mtime = os.path.getmtime(p)
            if p not in mtimes_before or new_mtime > mtimes_before[p]:
                with open(p, "rb") as f:
                    plot_b64 = base64.b64encode(f.read()).decode()
                break

    if plot_b64 is None:
        for p in plot_candidates:
            if p in answer and os.path.exists(p):
                with open(p, "rb") as f:
                    plot_b64 = base64.b64encode(f.read()).decode()
                break

    return jsonify({"answer": answer, "plot": plot_b64})


@app.route("/api/shutdown", methods=["POST"])
def shutdown():
    threading.Timer(0.25, lambda: os._exit(0)).start()
    return jsonify({"ok": True, "message": "shutting down"})


INDEX_HTML = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>VIPER Mission Planner</title>
<style>
  :root {
    --bg: #0f1117;
    --surface: #1a1d27;
    --border: #2a2d3a;
    --text: #e0e0e0;
    --muted: #8888a0;
    --accent: #6c8cff;
    --user-bg: #252940;
    --bot-bg: #1e2130;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    background: var(--bg);
    color: var(--text);
    height: 100vh;
    display: flex;
    flex-direction: column;
  }
  header {
    padding: 14px 24px;
    border-bottom: 1px solid var(--border);
    font-size: 16px;
    font-weight: 600;
    display: flex;
    align-items: center;
    gap: 10px;
  }
  header span.icon { font-size: 22px; }
  header .spacer { flex: 1; }
  #quit {
    background: transparent; border: 1px solid var(--border);
    color: var(--muted); padding: 5px 12px; border-radius: 6px;
    font-size: 12px; cursor: pointer;
  }
  #quit:hover { border-color: #ef5d6f; color: #ef5d6f; }
  #chat {
    flex: 1;
    overflow-y: auto;
    padding: 20px 24px;
    display: flex;
    flex-direction: column;
    gap: 16px;
  }
  .msg {
    max-width: 780px;
    width: 100%;
    margin: 0 auto;
    padding: 14px 18px;
    border-radius: 10px;
    line-height: 1.6;
    white-space: pre-wrap;
    font-size: 14px;
  }
  .msg.user {
    background: var(--user-bg);
    align-self: flex-end;
    border: 1px solid var(--border);
  }
  .msg.bot {
    background: var(--bot-bg);
    border: 1px solid var(--border);
  }
  .msg .label {
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    color: var(--muted);
    margin-bottom: 6px;
  }
  .msg img {
    max-width: 100%;
    margin-top: 12px;
    border-radius: 6px;
    border: 1px solid var(--border);
  }
  .typing {
    color: var(--muted);
    font-style: italic;
    max-width: 780px;
    width: 100%;
    margin: 0 auto;
    padding: 0 18px;
    font-size: 13px;
  }
  #input-bar {
    border-top: 1px solid var(--border);
    padding: 14px 24px;
    display: flex;
    gap: 10px;
    max-width: 840px;
    width: 100%;
    margin: 0 auto;
  }
  #input-bar input {
    flex: 1;
    padding: 12px 16px;
    border-radius: 8px;
    border: 1px solid var(--border);
    background: var(--surface);
    color: var(--text);
    font-size: 14px;
    outline: none;
  }
  #input-bar input:focus { border-color: var(--accent); }
  #input-bar button {
    padding: 10px 20px;
    border-radius: 8px;
    border: none;
    background: var(--accent);
    color: #fff;
    font-weight: 600;
    font-size: 14px;
    cursor: pointer;
  }
  #input-bar button:disabled { opacity: 0.5; cursor: default; }
  .hint {
    max-width: 780px;
    width: 100%;
    margin: 0 auto 8px;
    font-size: 12px;
    color: var(--muted);
    padding: 0 18px;
  }
</style>
</head>
<body>

<header>
  <span class="icon">&#127769;</span> VIPER Mission Planner
  <span class="spacer"></span>
  <button id="quit" type="button" title="Shut down the server">Quit</button>
</header>

<div id="chat">
  <div class="hint">
    Try: "run the simulation" &middot; "what are the axes of the plot?" &middot; "worst case scenario" &middot; "can we swing by (30, 45)?" &middot; "plan within a triangle"
  </div>
</div>

<form id="input-bar" autocomplete="off">
  <input id="q" type="text" placeholder="Ask about the mission..." autofocus />
  <button id="send" type="submit">Send</button>
</form>

<script>
const chat = document.getElementById("chat");
const qInput = document.getElementById("q");
const sendBtn = document.getElementById("send");
const form = document.getElementById("input-bar");

function addMsg(role, text, plotB64) {
  const div = document.createElement("div");
  div.className = "msg " + role;
  const label = document.createElement("div");
  label.className = "label";
  label.textContent = role === "user" ? "You" : "Assistant";
  div.appendChild(label);
  const body = document.createElement("div");
  body.textContent = text;
  div.appendChild(body);
  if (plotB64) {
    const img = document.createElement("img");
    img.src = "data:image/png;base64," + plotB64;
    img.alt = "Trajectory plot";
    div.appendChild(img);
  }
  chat.appendChild(div);
  chat.scrollTop = chat.scrollHeight;
}

let typing = null;
function showTyping() {
  typing = document.createElement("div");
  typing.className = "typing";
  typing.textContent = "Thinking...";
  chat.appendChild(typing);
  chat.scrollTop = chat.scrollHeight;
}
function hideTyping() {
  if (typing) { typing.remove(); typing = null; }
}

document.getElementById("quit").addEventListener("click", async () => {
  if (!confirm("Shut down the server?")) return;
  try { await fetch("/api/shutdown", { method: "POST" }); } catch (e) {}
  document.body.innerHTML =
    '<div style="padding:40px;font-family:sans-serif;color:#8888a0;">' +
    'Server stopped. You can close this tab.</div>';
});

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  const q = qInput.value.trim();
  if (!q) return;
  qInput.value = "";
  addMsg("user", q);
  sendBtn.disabled = true;
  showTyping();
  try {
    const res = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query: q }),
    });
    const data = await res.json();
    hideTyping();
    addMsg("bot", data.answer || data.error || "No response.", data.plot);
  } catch (err) {
    hideTyping();
    addMsg("bot", "Error: " + err.message);
  }
  sendBtn.disabled = false;
  qInput.focus();
});
</script>
</body>
</html>
"""

@app.route("/")
def index():
    return INDEX_HTML


if __name__ == "__main__":
    print("[Initializing DSPy agent...]")
    get_agent()
    url = "http://localhost:5001"
    print(f"Starting VIPER Mission Planner UI at {url}")
    threading.Timer(1.2, lambda: webbrowser.open(url)).start()
    app.run(host="127.0.0.1", port=5001, debug=False)
