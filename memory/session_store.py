import json
import uuid
from pathlib import Path

HISTORY_FILE = Path(__file__).parent / "session_history.txt"
class SessionStore:
    def __init__(self):
        self.session_id = None
        self.task = None
        self.environment = None
        self.history = []

    def new_session_id(self):
        return str(uuid.uuid4())[:8]

    def load_all(self):
        if not HISTORY_FILE.exists():
            return []
        try:
            return json.loads(HISTORY_FILE.read_text(encoding="utf-8"))
        except Exception:
            return []

    def save_session(self, session: dict):
        all_sessions = self.load_all()
        existing = next((s for s in all_sessions if s["session_id"] == session["session_id"]), None)
        if existing:
            existing.update(session)
        else:
            all_sessions.append(session)
        HISTORY_FILE.write_text(json.dumps(all_sessions, ensure_ascii=False, indent=2), encoding="utf-8")

    def delete_session(self, session_id):
        all_sessions = self.load_all()
        all_sessions = [s for s in all_sessions if s["session_id"] != session_id]
        HISTORY_FILE.write_text(json.dumps(all_sessions, ensure_ascii=False, indent=2), encoding="utf-8")

    def add_turn(self, session_id, task, environment, user_message, robot_message):
        all_sessions = self.load_all()
        session = next((s for s in all_sessions if s["session_id"] == session_id), None)
        if session is None:
            session = {
                "session_id": session_id,
                "task": task,
                "environment": environment,
                "history": [],
            }
            all_sessions.append(session)
        session["history"].append({"user_message": user_message, "robot_message": robot_message})
        HISTORY_FILE.write_text(json.dumps(all_sessions, ensure_ascii=False, indent=2), encoding="utf-8")
