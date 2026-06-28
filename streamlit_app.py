import os
import time
import httpx
import streamlit as st
from uuid import uuid4

API_URL = os.environ.get("BACKEND_URL", "http://localhost:8000") + "/chat"

st.set_page_config(
    page_title="NexusAI — Multi Agent Assistant",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── Base ── */
* { font-family: 'Inter', sans-serif; }
.stApp { background: #0a0a0f; color: #e2e8f0; }
.stApp > header { background: transparent; }

/* ── Hide default elements ── */
#MainMenu, footer, .stDeployButton { display: none !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: #0a0a0f; }
::-webkit-scrollbar-thumb { background: #6366f1; border-radius: 4px; }

/* ── Hero Header ── */
.hero-header {
    background: linear-gradient(135deg, #0f0f1a 0%, #1a0a2e 50%, #0a1628 100%);
    border: 1px solid rgba(99, 102, 241, 0.3);
    border-radius: 20px;
    padding: 32px 40px;
    margin-bottom: 24px;
    position: relative;
    overflow: hidden;
}
.hero-header::before {
    content: '';
    position: absolute;
    top: -50%;
    left: -50%;
    width: 200%;
    height: 200%;
    background: radial-gradient(circle at 30% 50%, rgba(99,102,241,0.08) 0%, transparent 60%),
                radial-gradient(circle at 70% 50%, rgba(139,92,246,0.06) 0%, transparent 60%);
    pointer-events: none;
}
.hero-title {
    font-size: 2.4rem;
    font-weight: 700;
    background: linear-gradient(135deg, #a78bfa, #6366f1, #38bdf8);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0 0 8px 0;
    letter-spacing: -0.5px;
}
.hero-subtitle {
    color: #64748b;
    font-size: 0.95rem;
    font-weight: 400;
    margin: 0;
}
.hero-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(99,102,241,0.15);
    border: 1px solid rgba(99,102,241,0.3);
    color: #a78bfa;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 500;
    margin-top: 12px;
}

/* ── Agent Pipeline ── */
.pipeline-container {
    background: linear-gradient(135deg, #0d0d1a, #0a1220);
    border: 1px solid rgba(99,102,241,0.2);
    border-radius: 16px;
    padding: 20px 24px;
    margin-bottom: 20px;
}
.pipeline-title {
    color: #475569;
    font-size: 0.7rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    margin-bottom: 16px;
}
.pipeline-flow {
    display: flex;
    align-items: center;
    gap: 0;
    flex-wrap: wrap;
}
.pipeline-node {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 6px;
}
.node-icon {
    width: 44px;
    height: 44px;
    border-radius: 12px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.1rem;
    border: 1px solid;
    transition: all 0.3s ease;
}
.node-icon.supervisor { background: rgba(99,102,241,0.15); border-color: rgba(99,102,241,0.4); }
.node-icon.research   { background: rgba(56,189,248,0.15); border-color: rgba(56,189,248,0.4); }
.node-icon.code       { background: rgba(52,211,153,0.15); border-color: rgba(52,211,153,0.4); }
.node-icon.writer     { background: rgba(251,146,60,0.15); border-color: rgba(251,146,60,0.4); }
.node-icon.critic     { background: rgba(239,68,68,0.15);  border-color: rgba(239,68,68,0.4); }
.node-icon.active {
    box-shadow: 0 0 20px currentColor;
    transform: scale(1.1);
}
.node-label {
    color: #475569;
    font-size: 0.65rem;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}
.pipeline-arrow {
    color: #1e293b;
    font-size: 1.2rem;
    margin: 0 4px;
    padding-bottom: 20px;
}

/* ── Active Route Badge ── */
.route-badge {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 8px 16px;
    border-radius: 24px;
    font-size: 0.82rem;
    font-weight: 600;
    margin-bottom: 12px;
    animation: fadeIn 0.4s ease;
}
.route-research { background: rgba(56,189,248,0.12); border: 1px solid rgba(56,189,248,0.35); color: #38bdf8; }
.route-code     { background: rgba(52,211,153,0.12); border: 1px solid rgba(52,211,153,0.35); color: #34d399; }
.route-writer   { background: rgba(251,146,60,0.12); border: 1px solid rgba(251,146,60,0.35); color: #fb923c; }
.route-unknown  { background: rgba(99,102,241,0.12); border: 1px solid rgba(99,102,241,0.35); color: #a78bfa; }

/* ── Answer Card ── */
.answer-card {
    background: linear-gradient(135deg, #0d0d1a, #0a0f1e);
    border: 1px solid rgba(99,102,241,0.2);
    border-radius: 16px;
    padding: 24px;
    margin: 8px 0;
    animation: slideUp 0.4s ease;
}
.answer-card:hover {
    border-color: rgba(99,102,241,0.4);
    box-shadow: 0 0 30px rgba(99,102,241,0.08);
    transition: all 0.3s ease;
}

/* ── Stats Row ── */
.stats-row {
    display: flex;
    gap: 12px;
    margin-top: 16px;
    flex-wrap: wrap;
}
.stat-chip {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    background: rgba(15,15,30,0.8);
    border: 1px solid rgba(99,102,241,0.15);
    color: #64748b;
    padding: 4px 10px;
    border-radius: 8px;
    font-size: 0.72rem;
    font-weight: 500;
}

/* ── User Message ── */
.user-bubble {
    background: linear-gradient(135deg, rgba(99,102,241,0.2), rgba(139,92,246,0.15));
    border: 1px solid rgba(99,102,241,0.3);
    border-radius: 16px 16px 4px 16px;
    padding: 14px 18px;
    color: #e2e8f0;
    font-size: 0.95rem;
    line-height: 1.6;
    margin: 4px 0;
    animation: fadeIn 0.3s ease;
}

/* ── Debug Expander ── */
.debug-content {
    background: #050508;
    border: 1px solid rgba(99,102,241,0.1);
    border-radius: 8px;
    padding: 12px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.75rem;
    color: #475569;
    white-space: pre-wrap;
    word-break: break-word;
    max-height: 200px;
    overflow-y: auto;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #07070f !important;
    border-right: 1px solid rgba(99,102,241,0.15) !important;
}
[data-testid="stSidebar"] * { color: #94a3b8 !important; }

.sidebar-section {
    background: rgba(99,102,241,0.06);
    border: 1px solid rgba(99,102,241,0.15);
    border-radius: 12px;
    padding: 14px;
    margin-bottom: 14px;
}
.sidebar-label {
    color: #475569 !important;
    font-size: 0.65rem !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 1.2px !important;
    margin-bottom: 8px !important;
}
.session-id {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.65rem;
    color: #6366f1 !important;
    word-break: break-all;
    line-height: 1.5;
}
.agent-info-item {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 8px 0;
    border-bottom: 1px solid rgba(99,102,241,0.08);
}
.agent-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    flex-shrink: 0;
}

/* ── Chat Input ── */
[data-testid="stChatInput"] {
    background: #0d0d1a !important;
    border: 1px solid rgba(99,102,241,0.3) !important;
    border-radius: 16px !important;
}
[data-testid="stChatInput"]:focus-within {
    border-color: rgba(99,102,241,0.6) !important;
    box-shadow: 0 0 20px rgba(99,102,241,0.1) !important;
}

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #4f46e5, #7c3aed) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    font-size: 0.82rem !important;
    padding: 8px 16px !important;
    transition: all 0.3s ease !important;
    width: 100% !important;
}
.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 8px 20px rgba(99,102,241,0.35) !important;
}

/* ── Spinner ── */
[data-testid="stSpinner"] { color: #6366f1 !important; }

/* ── Animations ── */
@keyframes fadeIn { from { opacity: 0; } to { opacity: 1; } }
@keyframes slideUp { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: translateY(0); } }
@keyframes pulse { 0%,100% { opacity:1; } 50% { opacity:0.5; } }
.pulse { animation: pulse 2s infinite; }
</style>
""", unsafe_allow_html=True)

# ── Session State ─────────────────────────────────────────────────────────────
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid4())
if "history" not in st.session_state:
    st.session_state.history = []
if "total_queries" not in st.session_state:
    st.session_state.total_queries = 0
if "route_counts" not in st.session_state:
    st.session_state.route_counts = {"research": 0, "code": 0, "writer": 0}

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 16px 0 20px 0;'>
        <div style='font-size:2rem; margin-bottom:6px;'>⚡</div>
        <div style='font-size:1.1rem; font-weight:700; background: linear-gradient(135deg,#a78bfa,#6366f1); -webkit-background-clip:text; -webkit-text-fill-color:transparent;'>NexusAI</div>
        <div style='font-size:0.7rem; color:#475569; margin-top:2px;'>Multi-Agent System</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sidebar-label">Session ID</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="sidebar-section"><div class="session-id">{st.session_state.thread_id}</div></div>', unsafe_allow_html=True)

    if st.button("⚡ New Conversation"):
        st.session_state.thread_id = str(uuid4())
        st.session_state.history = []
        st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="sidebar-label">Session Stats</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="sidebar-section">
        <div style='display:flex; justify-content:space-between; margin-bottom:8px;'>
            <span style='font-size:0.78rem;'>Total Queries</span>
            <span style='color:#a78bfa; font-weight:600;'>{st.session_state.total_queries}</span>
        </div>
        <div style='display:flex; justify-content:space-between; margin-bottom:8px;'>
            <span style='font-size:0.78rem;'>🔍 Research</span>
            <span style='color:#38bdf8; font-weight:600;'>{st.session_state.route_counts["research"]}</span>
        </div>
        <div style='display:flex; justify-content:space-between; margin-bottom:8px;'>
            <span style='font-size:0.78rem;'>💻 Code</span>
            <span style='color:#34d399; font-weight:600;'>{st.session_state.route_counts["code"]}</span>
        </div>
        <div style='display:flex; justify-content:space-between;'>
            <span style='font-size:0.78rem;'>✍️ Writer</span>
            <span style='color:#fb923c; font-weight:600;'>{st.session_state.route_counts["writer"]}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="sidebar-label">Agents</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="sidebar-section">
        <div class="agent-info-item">
            <div class="agent-dot" style="background:#6366f1;box-shadow:0 0 6px #6366f1;"></div>
            <div><div style='font-size:0.78rem;font-weight:500;'>Supervisor</div><div style='font-size:0.65rem;color:#475569;'>Routes queries</div></div>
        </div>
        <div class="agent-info-item">
            <div class="agent-dot" style="background:#38bdf8;box-shadow:0 0 6px #38bdf8;"></div>
            <div><div style='font-size:0.78rem;font-weight:500;'>Research</div><div style='font-size:0.65rem;color:#475569;'>Web search via Tavily</div></div>
        </div>
        <div class="agent-info-item">
            <div class="agent-dot" style="background:#34d399;box-shadow:0 0 6px #34d399;"></div>
            <div><div style='font-size:0.78rem;font-weight:500;'>Code</div><div style='font-size:0.65rem;color:#475569;'>Python execution</div></div>
        </div>
        <div class="agent-info-item">
            <div class="agent-dot" style="background:#fb923c;box-shadow:0 0 6px #fb923c;"></div>
            <div><div style='font-size:0.78rem;font-weight:500;'>Writer</div><div style='font-size:0.65rem;color:#475569;'>Prose & synthesis</div></div>
        </div>
        <div class="agent-info-item" style="border-bottom:none;">
            <div class="agent-dot" style="background:#ef4444;box-shadow:0 0 6px #ef4444;"></div>
            <div><div style='font-size:0.78rem;font-weight:500;'>Critic</div><div style='font-size:0.65rem;color:#475569;'>Quality gate</div></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ── Main Content ──────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-header">
    <div class="hero-title">⚡ NexusAI</div>
    <p class="hero-subtitle">Intelligent multi-agent system that routes your queries to specialized AI agents</p>
    <div class="hero-badge">🟢 <span class="pulse">Live</span> &nbsp;·&nbsp; LangGraph &nbsp;·&nbsp; Groq Llama 3.3 &nbsp;·&nbsp; Tavily Search</div>
</div>
""", unsafe_allow_html=True)

# ── Agent Pipeline Visual ─────────────────────────────────────────────────────
st.markdown("""
<div class="pipeline-container">
    <div class="pipeline-title">⚙ Agent Pipeline</div>
    <div class="pipeline-flow">
        <div class="pipeline-node">
            <div class="node-icon supervisor">🧠</div>
            <div class="node-label">Supervisor</div>
        </div>
        <div class="pipeline-arrow">→</div>
        <div class="pipeline-node">
            <div class="node-icon research">🔍</div>
            <div class="node-label">Research</div>
        </div>
        <div class="pipeline-arrow" style="color:#1e293b; font-size:0.7rem; padding-bottom:20px;">╱</div>
        <div class="pipeline-node">
            <div class="node-icon code">💻</div>
            <div class="node-label">Code</div>
        </div>
        <div class="pipeline-arrow" style="color:#1e293b; font-size:0.7rem; padding-bottom:20px;">╲</div>
        <div class="pipeline-node">
            <div class="node-icon writer">✍️</div>
            <div class="node-label">Writer</div>
        </div>
        <div class="pipeline-arrow">→</div>
        <div class="pipeline-node">
            <div class="node-icon critic">⚖️</div>
            <div class="node-label">Critic</div>
        </div>
        <div class="pipeline-arrow">→</div>
        <div class="pipeline-node">
            <div class="node-icon" style="background:rgba(52,211,153,0.15);border-color:rgba(52,211,153,0.4);">✅</div>
            <div class="node-label">Answer</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Helper: render response ───────────────────────────────────────────────────
ROUTE_CONFIG = {
    "research": ("🔍", "route-research", "Research Agent", "#38bdf8"),
    "code":     ("💻", "route-code",     "Code Agent",     "#34d399"),
    "writer":   ("✍️", "route-writer",   "Writer Agent",   "#fb923c"),
    "unknown":  ("🤖", "route-unknown",  "AI Agent",       "#a78bfa"),
}

def render_response(resp: dict, latency: float = 0):
    if "detail" in resp and "route" not in resp:
        st.markdown(f"""
        <div style='background:rgba(239,68,68,0.1);border:1px solid rgba(239,68,68,0.3);border-radius:12px;padding:16px;color:#f87171;'>
            ❌ <strong>Error:</strong> {resp['detail']}
        </div>
        """, unsafe_allow_html=True)
        return

    route = resp.get("route") or "unknown"
    icon, css_class, label, color = ROUTE_CONFIG.get(route, ROUTE_CONFIG["unknown"])
    retries = resp.get("retry_count", 0)
    answer = resp.get("answer", "No answer generated.")
    agent_outputs = resp.get("agent_outputs", {})

    # Route badge
    retry_text = f" · {retries} retr{'y' if retries==1 else 'ies'}" if retries > 0 else ""
    st.markdown(f"""
    <div class="route-badge {css_class}">
        {icon} {label}{retry_text}
        {'&nbsp;&nbsp;⚠️ <span style="font-size:0.72rem;">Quality checked</span>' if retries > 0 else ''}
    </div>
    """, unsafe_allow_html=True)

    # Answer card
    st.markdown(f'<div class="answer-card">', unsafe_allow_html=True)
    st.markdown(answer)
    st.markdown(f"""
    <div class="stats-row">
        <div class="stat-chip">⏱ {latency:.1f}s</div>
        <div class="stat-chip">🤖 {label}</div>
        <div class="stat-chip">🔄 {retries} retries</div>
        <div class="stat-chip">📦 {len(agent_outputs)} agent(s) used</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Debug expander
    if agent_outputs:
        with st.expander("🔬 View Agent Outputs", expanded=False):
            for agent, output in agent_outputs.items():
                agent_icon = {"research":"🔍","code":"💻","writer":"✍️","critic":"⚖️"}.get(agent,"🤖")
                st.markdown(f"**{agent_icon} {agent.title()} Agent**")
                st.markdown(f'<div class="debug-content">{output}</div>', unsafe_allow_html=True)

# ── Chat History ──────────────────────────────────────────────────────────────
for item in st.session_state.history:
    q, resp, lat = item if len(item) == 3 else (*item, 0)
    with st.chat_message("user", avatar="👤"):
        st.markdown(f'<div class="user-bubble">{q}</div>', unsafe_allow_html=True)
    with st.chat_message("assistant", avatar="⚡"):
        render_response(resp, lat)

# ── Input ─────────────────────────────────────────────────────────────────────
if len(st.session_state.history) == 0:
    st.markdown("""
    <div style='display:flex; gap:10px; flex-wrap:wrap; margin:16px 0;'>
        <div style='background:rgba(56,189,248,0.08);border:1px solid rgba(56,189,248,0.2);border-radius:10px;padding:10px 14px;font-size:0.8rem;color:#38bdf8;cursor:pointer;'>🔍 What is quantum computing?</div>
        <div style='background:rgba(52,211,153,0.08);border:1px solid rgba(52,211,153,0.2);border-radius:10px;padding:10px 14px;font-size:0.8rem;color:#34d399;cursor:pointer;'>💻 Calculate 15 factorial</div>
        <div style='background:rgba(251,146,60,0.08);border:1px solid rgba(251,146,60,0.2);border-radius:10px;padding:10px 14px;font-size:0.8rem;color:#fb923c;cursor:pointer;'>✍️ Write a poem about AI</div>
    </div>
    """, unsafe_allow_html=True)

query = st.chat_input("Ask NexusAI anything…")
if query:
    with st.chat_message("user", avatar="👤"):
        st.markdown(f'<div class="user-bubble">{query}</div>', unsafe_allow_html=True)

    with st.chat_message("assistant", avatar="⚡"):
        with st.spinner("🧠 Agents processing…"):
            start = time.time()
            try:
                http_resp = httpx.post(
                    API_URL,
                    json={"query": query, "thread_id": st.session_state.thread_id},
                    timeout=120,
                )
                resp = http_resp.json()
                latency = time.time() - start
            except httpx.ConnectError:
                st.markdown("""
                <div style='background:rgba(239,68,68,0.1);border:1px solid rgba(239,68,68,0.3);border-radius:12px;padding:16px;color:#f87171;'>
                    ❌ <strong>Backend offline.</strong> Start with: <code>uvicorn app:app --reload</code>
                </div>
                """, unsafe_allow_html=True)
                st.stop()
            except Exception as e:
                st.error(f"❌ {e}")
                st.stop()

        render_response(resp, latency)

        # Update stats
        route = resp.get("route", "")
        if route in st.session_state.route_counts:
            st.session_state.route_counts[route] += 1
        st.session_state.total_queries += 1

    st.session_state.history.append((query, resp, latency))
