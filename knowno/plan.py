import json
import re
import sys
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))


class TaskPlanner:
    """
    Lập plan (chuỗi bước thực hiện) từ ambiguous_task.

    Workflow:
    1. Dùng EnvironmentMatcher để tự động tìm environment phù hợp từ dataset.
    2. Gọi LLM với system prompt từ plan.txt + user message chứa task & environment.
    3. Parse JSON response → trả về list các step dict.
    """

    DEFAULT_DATASET = _root / "ambik_dataset" / "ambik_test_900.csv"
    DEFAULT_PROMPT = Path(__file__).parent / "prompts" / "plan.txt"

    def __init__(self, llm_instance, dataset_path=None, prompt_path=None):
        """
        :param llm_instance: Instance của LLM (Ollama).
        :param dataset_path: Đường dẫn tới CSV dataset (mặc định: ambik_test_900.csv).
        :param prompt_path: Đường dẫn tới file prompt (mặc định: prompts/plan.txt).
        """
        from knowno.embedding import EnvironmentMatcher

        self.llm = llm_instance
        self.env_matcher = EnvironmentMatcher(dataset_path or self.DEFAULT_DATASET)
        self.system_prompt = self._load_prompt(prompt_path or self.DEFAULT_PROMPT)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_prompt(self, prompt_path):
        try:
            return Path(prompt_path).read_text(encoding="utf-8").strip()
        except Exception as e:
            print(f"[Warning] Không thể đọc prompt từ {prompt_path}: {e}")
            return (
                "You are a Kitchen Robot Task Planner. "
                "Decompose the User Task into atomic robotic steps. "
                "Return ONLY a valid JSON array."
            )

    def _build_user_message(self, task: str, environment: list[str]) -> str:
        if environment is None:
            env_str = "Not provided"
        else:
            env_str = ", ".join(environment) if environment else "(empty)"
        return (
            f"User Task: {task}\n"
            f"Environment: {env_str}\n\n"
            "Generate the final plan now."
        )

    def _parse_steps(self, response_text: str) -> list[dict]:
        """Parse plan từ LLM response (ưu tiên numbered list, fallback JSON)."""
        text = response_text.strip()

        # Ưu tiên parse numbered list:
        # 1. ...
        # 2. ...
        numbered_lines = re.findall(r"^\s*\d+\.\s+(.+?)\s*$", text, flags=re.MULTILINE)
        if numbered_lines:
            return [
                {"step_id": idx, "action": "plan_step", "target_object": line.strip()}
                for idx, line in enumerate(numbered_lines, start=1)
            ]

        # Bóc markdown code block nếu có
        md_match = re.search(r"```(?:json)?\s*([\s\S]+?)```", text)
        if md_match:
            text = md_match.group(1).strip()

        # Thử parse thẳng
        try:
            steps = json.loads(text)
            if isinstance(steps, list):
                return steps
        except json.JSONDecodeError:
            pass

        # Tìm JSON array đầu tiên trong text
        arr_match = re.search(r"\[[\s\S]+\]", text)
        if arr_match:
            try:
                steps = json.loads(arr_match.group())
                if isinstance(steps, list):
                    return steps
            except json.JSONDecodeError:
                pass

        print("[Warning] Không parse được JSON plan từ LLM response.")
        return []

    def _validate_steps(self, steps: list, environment: list[str]) -> list[dict]:
        """
        Đảm bảo mỗi step có đủ các trường cần thiết.
        Không loại bỏ bước nào — chỉ đảm bảo schema hợp lệ.
        """
        validated = []
        for i, step in enumerate(steps, start=1):
            if not isinstance(step, dict):
                continue
            validated.append(
                {
                    "step_id": step.get("step_id", i),
                    "action": str(step.get("action", "")).strip(),
                    "target_object": str(step.get("target_object", "")).strip(),
                }
            )
        return validated

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def plan(self, ambiguous_task: str) -> dict:
        """
        Tự động tìm environment và lập plan cho ambiguous_task.

        :param ambiguous_task: Chuỗi mô tả task (có thể mơ hồ).
        :return: Dict gồm:
            {
                "task": str,
                "environment": list[str],
                "steps_with_env": list[{"step_id", "action", "target_object"}],
                "steps_without_env": list[{"step_id", "action", "target_object"}]
            }
        """
        # Bước 1: Tìm environment từ dataset
        environment = self.env_matcher.find_environment(ambiguous_task)
        if not environment:
            print(f"[Warning] Không tìm thấy environment cho task: {ambiguous_task!r}")

        # Bước 2a: Gọi LLM CÓ environment
        user_msg_with = self._build_user_message(ambiguous_task, environment)
        try:
            res_with = self.llm.chat(self.system_prompt, user_msg_with)
        except Exception as e:
            print(f"[Error] LLM call thất bại (với environment): {e}")
            res_with = "[]"

        # Bước 2b: Gọi LLM KHÔNG CÓ environment
        user_msg_without = self._build_user_message(ambiguous_task, None)
        try:
            res_without = self.llm.chat(self.system_prompt, user_msg_without)
        except Exception as e:
            print(f"[Error] LLM call thất bại (không có environment): {e}")
            res_without = "[]"

        # Bước 3: Parse & validate
        raw_steps_with = self._parse_steps(res_with)
        steps_with_env = self._validate_steps(raw_steps_with, environment)

        raw_steps_without = self._parse_steps(res_without)
        steps_without_env = self._validate_steps(raw_steps_without, [])

        return {
            "task": ambiguous_task,
            "environment": environment,
            "steps_with_env": steps_with_env,
            "steps_without_env": steps_without_env,
        }


# ----------------------------------------------------------------------
# Chạy thử CLI: python3 knowno/plan.py
# ----------------------------------------------------------------------
if __name__ == "__main__":
    from llm import LLM

    llm = LLM("ollama:qwen2.5:32b", {"max_new_tokens": 512, "temperature": 0.2})
    planner = TaskPlanner(llm)

    print("=== Kitchen Robot Task Planner ===")
    print("(Nhập task hoặc Ctrl+C để thoát)\n")

    while True:
        try:
            task = input("Ambiguous task: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nBye!")
            break

        if not task:
            continue

        result = planner.plan(task)

        print(f"\nEnvironment found ({len(result['environment'])} items):")
        print("  " + ", ".join(result["environment"]))

        print(f"\n[1] Plan WITH environment ({len(result['steps_with_env'])} steps):")
        for step in result["steps_with_env"]:
            if step.get("action") == "plan_step":
                print(f"  {step['step_id']:>2}. {step['target_object']}")
            else:
                print(
                    f"  {step['step_id']:>2}. [{step['action']}]  →  {step['target_object']}"
                )

        print(f"\n[2] Plan WITHOUT environment ({len(result['steps_without_env'])} steps):")
        for step in result["steps_without_env"]:
            if step.get("action") == "plan_step":
                print(f"  {step['step_id']:>2}. {step['target_object']}")
            else:
                print(
                    f"  {step['step_id']:>2}. [{step['action']}]  →  {step['target_object']}"
                )
        print()
