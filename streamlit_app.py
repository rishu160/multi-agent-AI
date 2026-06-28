import os
import httpx
import streamlit as st
from uuid import uuid4

API_URL = os.environ.get("BACKEND_URL", "http://localhost:8000") + "/chat"
AGENT_COLORS = {"research": "🔍", "code": "💻", "writer": "✍️"}

st.set_page_config(page_title="Multi-Agent Assistant", page_icon="🤖")
st.title("🤖 Multi-Agent AI Assistant")
st.caption("Powered by LangGraph + Claude — routes queries to Research, Code, or Writer agents")

# ── Session state ─────────────────────────────────────────────────────────────
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid4())
if "history" not in st.session_state:
    st.session_state.history = []


def render_response(resp: dict):
    """Render a response dict safely — works even if backend returned an error."""
    # If backend returned an HTTP error, resp will have "detail" not "route"
    if "detail" in resp and "route" not in resp:
        st.error(f"Backend error: {resp['detail']}")
        return

    route = resp.get("route") or "unknown"
    icon = AGENT_COLORS.get(route, "🤖")
    st.caption(
        f"{icon} Routed to **{route.title()} Agent** · "
        f"retries: {resp.get('retry_count', 0)}"
    )
    st.markdown(resp.get("answer", "No answer generated."))
    with st.expander("Agent outputs (debug)"):
        for agent, output in resp.get("agent_outputs", {}).items():
            st.markdown(f"**{agent.title()} Agent**")
            st.text(output)


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Session")
    st.code(st.session_state.thread_id, language=None)
    if st.button("New conversation"):
        st.session_state.thread_id = str(uuid4())
        st.session_state.history = []
        st.rerun()

# ── Chat history ──────────────────────────────────────────────────────────────
for query, resp in st.session_state.history:
    with st.chat_message("user"):
        st.write(query)
    with st.chat_message("assistant"):
        render_response(resp)

# ── Input ─────────────────────────────────────────────────────────────────────
query = st.chat_input("Ask anything…")
if query:
    with st.chat_message("user"):
        st.write(query)

    with st.chat_message("assistant"):
        with st.spinner("Agents thinking…"):
            try:
                http_resp = httpx.post(
                    API_URL,
                    json={"query": query, "thread_id": st.session_state.thread_id},
                    timeout=120,
                )
                resp = http_resp.json()
            except httpx.ConnectError:
                st.error("❌ Cannot connect to backend. Make sure `uvicorn app:app --reload` is running in another terminal.")
                st.stop()
            except Exception as e:
                st.error(f"❌ Request failed: {e}")
                st.stop()

        render_response(resp)

    st.session_state.history.append((query, resp))
