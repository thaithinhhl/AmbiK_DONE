import redis
import json
import uuid


class ChatHistory:
    def __init__(self, host: str = "localhost", port: int = 6379, db: int = 0):
        self.redis = redis.Redis(host=host, port=port, db=db)
        self.current_session_id = None

    def start_session(self):
        self.current_session_id = str(uuid.uuid4())[:8]
        key = f"session:{self.current_session_id}"

        initial_data = {
            "session_id": self.current_session_id,
            "task": None,
            "environment": [],
            "history": [],
        }

        self.redis.set(key, json.dumps(initial_data), ex=3600)
        return initial_data

    def set_task_context(self, task, environment):
        if not self.current_session_id:
            return False

        key = f"session:{self.current_session_id}"
        raw = self.redis.get(key)
        if raw is None:
            return False

        data = json.loads(raw)
        data["task"] = task
        data["environment"] = environment
        self.redis.set(key, json.dumps(data), ex=3600)
        return True

    def get_task_context(self):
        if not self.current_session_id:
            return None, []

        key = f"session:{self.current_session_id}"
        raw = self.redis.get(key)
        if raw is None:
            return None, []

        data = json.loads(raw)
        return data.get("task"), data.get("environment", [])

    def add_message(self, user_message, robot_message):
        if not self.current_session_id:
            return False

        key = f"session:{self.current_session_id}"
        raw = self.redis.get(key)
        if raw is None:
            return False

        data = json.loads(raw)

        chat_turn = {
            "user": user_message,
            "robot_kitchen": robot_message,
        }

        data["history"].append(chat_turn)
        self.redis.set(key, json.dumps(data), ex=3600)
        return True

    def end_session(self):
        if self.current_session_id:
            key = f"session:{self.current_session_id}"
            raw = self.redis.get(key)
            history_data = json.loads(raw) if raw else None
            self.redis.delete(key)
            temp_id = self.current_session_id
            self.current_session_id = None
            return temp_id, history_data
        return None, None
