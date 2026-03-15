import sys
sys.dont_write_bytecode = True

import re
import html as _html
import base64
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import streamlit as st
from llm import LLM
from knowno.pipeline import EntityExtractor, TaskHandler, ResponseGenerator
from knowno.embedding import EnvironmentMatcher, EmbeddingSelector
from knowno.classify import AmbiguityClassifier
from memory.session_store import SessionStore

DATASET_PATH = Path(__file__).parent / "ambik_dataset" / "ambik_test_900.csv"
GREETING = "hey, robot kitchen"
FAREWELL = "thank you, robot kitchen"
BOT_NAME = "Robot Kitchen"


def _img_b64(path: Path) -> str:
    try:
        return base64.b64encode(path.read_bytes()).decode("ascii")
    except Exception:
        return ""


ROBOT_AVATAR_B64 = _img_b64(Path(__file__).with_name("robot.png"))
USER_AVATAR_B64 = _img_b64(Path(__file__).with_name("user.png"))

if __name__ == "__main__":
    st.set_page_config(page_title="Robot Kitchen", page_icon="🤖", layout="centered")

    # ── CSS ──────────────────────────────────────────────────────────────────
    st.markdown("""
<style>
/* Page background - tím nhạt như cũ */
.stApp {
  background: radial-gradient(900px 500px at 20% 0%, #f3e8ff 0%, rgba(243,232,255,0) 60%),
              radial-gradient(900px 500px at 80% 0%, #ede9fe 0%, rgba(237,233,254,0) 60%),
              linear-gradient(180deg, #f5f0ff 0%, #ede9fe 100%);
}
header[data-testid="stHeader"] { background: transparent; }
#MainMenu, footer { visibility: hidden; }
.block-container { padding-top: 2rem !important; max-width: 480px !important; }

/* Remove Streamlit default border wrappers */
[data-testid="stVerticalBlockBorderWrapper"] { border: none !important; background: transparent !important; }

/* Ẩn st.chat_message mặc định */
[data-testid="stChatMessage"] { display: none !important; }

/* Widget card */
.kb-widget {
  background: #ffffff;
  border-radius: 16px;
  overflow: hidden;
  box-shadow: 0 6px 32px rgba(0,0,0,0.13);
  max-width: 440px;
  margin: 0 auto;
}

/* Header */
.kb-header {
  background: #7c3aed;
  padding: 14px 18px;
  display: flex;
  align-items: center;
  gap: 12px;
}
.kb-avatar {
  width: 38px; height: 38px;
  background: rgba(255,255,255,0.2);
  border-radius: 50%;
  overflow: hidden;
  display: flex; align-items: center; justify-content: center;
  flex-shrink: 0;
}
.kb-info { flex: 1; }
.kb-name  { color: #ffffff; font-weight: 700; font-size: 15px; }
.kb-status { color: rgba(255,255,255,0.85); font-size: 12px; margin-top: 1px; }

/* Chat area */
.kb-chat-area {
  background: #ffffff;
  padding: 16px 14px 8px;

  height: 560px;
  overflow-y: auto;
  border-radius: 16px;
  margin-top: 18px;
  box-shadow: 0 6px 20px rgba(0,0,0,0.08);
}

/* Bot row */
.kb-bot-row {
  display: flex; align-items: flex-start;
  gap: 10px; margin: 12px 0 4px;
}
.kb-bot-icon {
  width: 36px; height: 36px; flex-shrink: 0;
  background: #7c3aed; border-radius: 50%;
  overflow: hidden;
  display: flex; align-items: center; justify-content: center;
}

.kb-bot-body { display: flex; flex-direction: column; max-width: 74%; }
.kb-bot-label { font-size: 11px; color: #9ca3af; margin-bottom: 4px; font-weight: 500; }
.kb-bot-bubble {
  background: #f3f4f6;
  color: #111827;
  padding: 10px 14px;
  border-radius: 4px 18px 18px 18px;
  font-size: 14px; line-height: 1.5;
  word-break: break-word;
}

/* User row */
.kb-user-row {
  display: flex; justify-content: flex-end; align-items: flex-end;
  margin: 8px 0;
}
.kb-user-bubble {
  background: #7c3aed;
  color: #ffffff;
  padding: 10px 16px;
  border-radius: 18px 18px 4px 18px;
  font-size: 14px; line-height: 1.5;
  max-width: 74%;
  word-break: break-word;
  box-shadow: 0 2px 8px rgba(124,58,237,0.3);
}

.kb-user-icon {
  width: 32px; height: 32px;
  margin-left: 8px;
  border-radius: 50%;
  overflow: hidden;
  flex-shrink: 0;
  background: #e5e7eb;
}

/* Quick-reply pills */
.kb-quick-row {
  display: flex; flex-wrap: wrap;
  gap: 8px; margin: 6px 0 10px 46px;
}
.kb-quick-pill {
  border: 1.5px solid #7c3aed;
  color: #7c3aed;
  background: #ffffff;
  border-radius: 20px;
  padding: 6px 16px;
  font-size: 13px; cursor: pointer;
  transition: background .15s;
}
.kb-quick-pill:hover { background: #f5f3ff; }

/* Input area */
.kb-input-area {
  background: #f9fafb;
  border-top: 1px solid #e5e7eb;
  padding: 10px 14px;
  border-radius: 0 0 16px 16px;
}

/* st.chat_input restyling */
.stChatInputContainer {
  background: #f9fafb !important;
  border: none !important;
  border-radius: 0 0 16px 16px !important;
  padding: 0 !important;
}
.stChatInputContainer textarea {
  background: #f9fafb !important;
  color: #374151 !important;
  -webkit-text-fill-color: #374151 !important;
  border: none !important;
  border-radius: 0 !important;
  font-size: 14px !important;
}
.stChatInputContainer textarea::placeholder { color: #9ca3af !important; }

/* All Streamlit buttons: base style */
.stButton > button {
  background: #7c3aed !important;
  border: none !important;
  color: #ffffff !important;
  -webkit-text-fill-color: #ffffff !important;
  border-radius: 8px !important;
  padding: 4px 10px !important;
  font-size: 15px !important;
  min-height: 0 !important;
  font-weight: 600 !important;
}
.stButton > button:hover { background: #6d28d9 !important; }

/* Quick-reply pills — scoped by class on parent container */
.quick-reply-row .stButton > button {
  background: #ffffff !important;
  border: 1.5px solid #7c3aed !important;
  color: #7c3aed !important;
  -webkit-text-fill-color: #7c3aed !important;
  border-radius: 20px !important;
  font-size: 13px !important;
  padding: 6px 14px !important;
  font-weight: 500 !important;
}
.quick-reply-row .stButton > button:hover {
  background: #f5f3ff !important;
}

/* New session button */
.new-session-btn .stButton > button {
  width: 100%;
  border-radius: 12px !important;
  padding: 10px !important;
  font-size: 14px !important;
}
</style>
""", unsafe_allow_html=True)

    # ── Pipeline ─────────────────────────────────────────────────────────────
    @st.cache_resource
    def load_pipeline_cached():
        llm = LLM("ollama:qwen2.5:32b", {"max_new_tokens": 150, "temperature": 0.7})
        embed_model = LLM("ollama:mxbai-embed-large:latest", {})
        extractor = EntityExtractor(llm)
        env_matcher = EnvironmentMatcher(DATASET_PATH)
        embedding_selector = EmbeddingSelector(embed_model)
        classifier = AmbiguityClassifier(llm)
        responder = ResponseGenerator(llm)
        handler = TaskHandler(extractor, env_matcher, embedding_selector, classifier)
        return handler, responder

    handler, responder = load_pipeline_cached()

    # ── State ────────────────────────────────────────────────────────────────
    def init_state():
        defaults = {
            "stage": "idle",
            "session_id": None,
            "task": None,
            "messages": [],
            "session_store": None,
            "clarification_pending": False,
            "pending_step": None,
            "pending_entities": None,
            "pending_top_objects": None,
            "conversation_history": [],
            "pending_input": None,
            "minimized": False,
            "last_viable_objects": [],
        }
        for k, v in defaults.items():
            if k not in st.session_state:
                st.session_state[k] = v

    init_state()
    if st.session_state.session_store is None:
        st.session_state.session_store = SessionStore()

    def add_message(role, content, meta=None):
        st.session_state.messages.append({"role": role, "content": content, "meta": meta})

    # ── HTML helpers ─────────────────────────────────────────────────────────
    def _md(text: str) -> str:
        text = _html.escape(text)
        text = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', text)
        text = re.sub(r'\*(.+?)\*',    r'<em>\1</em>', text)
        text = re.sub(r'`([^`]+)`',
            r'<code style="background:rgba(0,0,0,0.06);padding:1px 4px;border-radius:3px;">\1</code>', text)
        text = text.replace('\n', '<br>')
        return text

    def bot_html(text: str) -> str:
        return f"""
<div class="kb-bot-row">
  <div class="kb-bot-icon">
    <img src="data:image/png;base64,{ROBOT_AVATAR_B64}" style="width:100%;height:100%;object-fit:cover;display:block;" />
  </div>
  <div class="kb-bot-body">
    <div class="kb-bot-label">{BOT_NAME}</div>
    <div class="kb-bot-bubble">{_md(text)}</div>
  </div>
</div>"""

    def user_html(text: str) -> str:
        return f"""
<div class="kb-user-row">
  <div class="kb-user-bubble">{_md(text)}</div>
  <div class="kb-user-icon">
    <img src="data:image/png;base64,{USER_AVATAR_B64}" style="width:100%;height:100%;object-fit:cover;display:block;" />
  </div>
</div>"""

    def render_chat_html(include_thinking: bool = False) -> str:
        """Build the entire scrollable chat area as one HTML string."""
        parts = []
        for msg in st.session_state.messages:
            if msg["role"] == "user":
                parts.append(user_html(msg["content"]))
            else:
                parts.append(bot_html(msg["content"]))
        if include_thinking:
            parts.append(bot_html("⏳ Thinking..."))
        inner = "\n".join(parts)
        return f"""
<div class="kb-chat-area" id="kb-chat-scroll">{inner}</div>
<script>
(function(){{
  var d = document.getElementById('kb-chat-scroll');
  if (d) d.scrollTop = d.scrollHeight;
}})();
</script>"""

    # ── Process pending input ─────────────────────────────────────────────────
    def process_pending():
        user_input = st.session_state.pending_input
        stage = st.session_state.stage
        store = st.session_state.session_store

        if stage == "idle":
            if user_input.strip().lower() == GREETING:
                st.session_state.session_id = store.new_session_id()
                st.session_state.stage = "awaiting_task"
                add_message("assistant",
                    f"Hi! I'm {BOT_NAME}.\nWhat task would you like me to do?")
            else:
                add_message("assistant", f'Please say **"{GREETING}"** to begin.')

        elif stage == "awaiting_task":
            task = user_input.strip()
            handler.start_task(task)
            st.session_state.task = task
            store.save_session({
                "session_id": st.session_state.session_id,
                "task": task,
                "environment": handler.environment,
                "history": [],
            })
            st.session_state.stage = "chatting"
            add_message("assistant",
                f"Got it! Task: **{task}**\nTell me the steps. Say *\"{FAREWELL}\"* when done.")

        elif stage == "chatting":
            if user_input.strip().lower() == FAREWELL:
                store.delete_session(st.session_state.session_id)
                st.session_state.stage = "ended"
                st.session_state.last_viable_objects = []
                add_message("assistant", "You're welcome! Session ended. Goodbye! 👋")
                st.session_state.pending_input = None
                return

            was_clarification = st.session_state.clarification_pending
            original_pending_step = st.session_state.pending_step

            if was_clarification:
                result = handler.clarify_step(
                    user_input,
                    original_pending_step,
                    st.session_state.pending_entities,
                    st.session_state.pending_top_objects,
                    history=st.session_state.conversation_history,
                )
                # Reset clarification state trước khi xử lý kết quả
                st.session_state.clarification_pending = False
                st.session_state.pending_step = None
                st.session_state.pending_entities = None
                st.session_state.pending_top_objects = None
            else:
                result = handler.handle_step(user_input, history=st.session_state.conversation_history)

            entities = result.get("entities", {})
            actions  = entities.get("actions", []) if isinstance(entities, dict) else []
            cls      = result.get("classification")

            robot_response = ""
            meta = {
                "actions":     actions,
                "objects":     entities.get("objects", []) if isinstance(entities, dict) else [],
                "top_objects": [(o, round(s, 4)) for o, s in result.get("top_objects", [])],
            }

            if cls:
                action_str     = ", ".join(actions)
                robot_response = responder.generate(
                    user_input, cls, action_str,
                    history=st.session_state.conversation_history,
                )
                meta["classification"]  = cls.get("status")
                meta["label"]           = cls.get("label")
                meta["viable_objects"]  = cls.get("viable_objects", [])

                if cls.get("status") == "Ambiguous":
                    st.session_state.clarification_pending = True
                    # Nếu vừa clarify mà vẫn Ambiguous → dùng combined step gốc để tiếp tục
                    if was_clarification and original_pending_step:
                        st.session_state.pending_step = f"{original_pending_step} [User specified: {user_input}]"
                    else:
                        st.session_state.pending_step = user_input
                    st.session_state.pending_entities    = entities
                    st.session_state.pending_top_objects = result.get("top_objects", [])
                    st.session_state.last_viable_objects = cls.get("viable_objects", [])
                else:
                    st.session_state.last_viable_objects = []

            final_response = robot_response or "Done. What's next?"
            st.session_state.conversation_history.append(
                {"user_message": user_input, "robot_message": final_response}
            )
            st.session_state.conversation_history = st.session_state.conversation_history[-3:]
            store.add_turn(
                st.session_state.session_id, st.session_state.task,
                handler.environment, user_input, final_response,
            )
            add_message("assistant", final_response, meta)

        st.session_state.pending_input = None

    # ── Render widget ─────────────────────────────────────────────────────────
    # Header: full-width purple bar with action buttons inside
    col_info, col_btns = st.columns([0.72, 0.28], vertical_alignment="center")
    with col_info:
        st.markdown(f"""
<div class="kb-header">
  <div class="kb-avatar">
    <img src="data:image/png;base64,{ROBOT_AVATAR_B64}" style="width:100%;height:100%;object-fit:cover;display:block;" />
  </div>
  <div class="kb-info">
    <div class="kb-name">{BOT_NAME}</div>
    <div class="kb-status">🟢 Online Now</div>
  </div>
</div>""", unsafe_allow_html=True)
    with col_btns:
        b1, b2 = st.columns(2)
        with b1:
            if st.button("↻", key="btn_refresh", help="Clear chat"):
                st.session_state.messages = []
                st.session_state.conversation_history = []
                st.session_state.last_viable_objects = []
                st.session_state.clarification_pending = False
                st.session_state.pending_step = None
                st.session_state.pending_entities = None
                st.session_state.pending_top_objects = None
                st.session_state.pending_input = None
                st.rerun()
        with b2:
            label = "—" if not st.session_state.minimized else "□"
            if st.button(label, key="btn_min", help="Minimize"):
                st.session_state.minimized = not st.session_state.minimized
                st.rerun()

    if st.session_state.minimized:
        st.caption("Chat minimized — click □ to expand.")
    else:
        # ── Chat frame: ONE single st.markdown call keeps all bubbles inside ──
        if st.session_state.pending_input is not None:
            # Show messages + thinking bubble, then process
            st.markdown(render_chat_html(include_thinking=True), unsafe_allow_html=True)
            process_pending()
            st.rerun()
        else:
            st.markdown(render_chat_html(include_thinking=False), unsafe_allow_html=True)

        # Quick-reply pills (viable objects when ambiguous)
        viable = st.session_state.last_viable_objects
        if viable and st.session_state.clarification_pending:
            st.markdown('<div class="quick-reply-row">', unsafe_allow_html=True)
            cols = st.columns(len(viable))
            for i, obj in enumerate(viable):
                with cols[i]:
                    if st.button(obj, key=f"vobj_{i}", use_container_width=True):
                        add_message("user", obj)
                        st.session_state.pending_input = obj
                        st.session_state.last_viable_objects = []
                        st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)

        # Input + session footer
        stage = st.session_state.stage
        if stage != "ended":
            placeholder = {
                "idle":          f'Say "{GREETING}"...',
                "awaiting_task": "Describe the task...",
                "chatting":      "Reply to Robot Kitchen...",
            }.get(stage, "")
            if user_input := st.chat_input(placeholder):
                add_message("user", user_input)
                st.session_state.pending_input = user_input
                st.session_state.last_viable_objects = []
                st.rerun()

            # Session id nhỏ phía dưới khung text
            if st.session_state.session_id:
                st.caption(f"Session ID: `{st.session_state.session_id}`")
        else:
            with st.container():
                st.markdown('<div class="new-session-btn">', unsafe_allow_html=True)
                if st.button("🔄 Start new session", use_container_width=True):
                    for k in list(st.session_state.keys()):
                        del st.session_state[k]
                    st.rerun()
                st.markdown('</div>', unsafe_allow_html=True)
