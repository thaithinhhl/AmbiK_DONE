import json
import re
import sys
from pathlib import Path

# Ensure project root is on path when running this script directly
_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    
    sys.path.insert(0, str(_root))

from llm import LLM
from knowno.embedding import EnvironmentMatcher, EmbeddingSelector
from knowno.classify import AmbiguityClassifier
from knowno.plan import TaskPlanner
from memory.session_store import SessionStore

class EntityExtractor:    
    def __init__(self, llm_instance):
        """
        :param llm_instance: Instance của lớp LLM (Ollama) đã thiết lập trước đó.
        """
        self.llm = llm_instance

    def extract(self, query):
        """
        Trích xuất cả hành động (actions) và vật thể (objects) từ câu lệnh.
        return Dict với 2 key:
                 {
                    "actions": [...],
                    "objects": [...]
                 }
        """
        prompt = f"""
        [ROLE]
        You are a semantic parser for a kitchen robot. 
        Your job is to extract:
        - actions: the operations / verbs the robot must do
        - objects: the physical objects / nouns (kitchen items, tools, food)

        [TASK]
        Query: "{query}"

        [OUTPUT FORMAT]
        Return ONLY a JSON object with exactly 2 keys:
        {{
          "actions": ["...", "..."],
          "objects": ["...", "..."]
        }}

        - Do not add any extra text.
        - If nothing is found for a field, return an empty array for that field.

        JSON:
        """
        
        response_text = self.llm.generate(prompt)
        
        try:
            # Tìm JSON object {...} cuối cùng trong response
            json_matches = list(re.finditer(r'\{.*?\}', response_text, re.DOTALL))
            if json_matches:
                json_str = json_matches[-1].group()
                parsed = json.loads(json_str)
            else:
                parsed = json.loads(response_text)
            
            actions = parsed.get("actions", []) if isinstance(parsed, dict) else []
            objects = parsed.get("objects", []) if isinstance(parsed, dict) else []
            
            if not isinstance(actions, list):
                actions = [actions] if actions else []
            if not isinstance(objects, list):
                objects = [objects] if objects else []
            
            actions_norm = [str(a).lower().strip() for a in actions if str(a).strip()]
            objects_norm = [str(o).lower().strip() for o in objects if str(o).strip()]
            
            return {
                "actions": actions_norm,
                "objects": objects_norm
            }
        except Exception as e:
            print(f"[Error] Không thể parse JSON từ LLM (actions/objects): {e}")
            return {
                "actions": [],
                "objects": []
            }


class TaskHandler:
    """
    Xử lý task theo hai giai đoạn:
    - task: nhập 1 lần duy nhất, chỉ lưu, không extract.
    - query: mỗi bước (handle_step) mới gọi extract.
    """

    def __init__(self, extractor, env_matcher, embedding_selector, classifier=None):
        self.extractor = extractor
        self.env_matcher = env_matcher
        self.embedding_selector = embedding_selector
        self.classifier = classifier
        self.current_task = None  # Chỉ set 1 lần qua start_task(task)
        self.environment = None   # Được set trong start_task dựa trên task
        self.entities = None      # Được set mỗi lần handle_step(query)
        self.planned_steps = []
        self.plan_source = None

    def start_task(self, task):
        """
        Giai đoạn 1: Tiếp nhận task tổng quát (chỉ gọi 1 lần).
        Không thực hiện extract, chỉ lưu task và tìm environment tương ứng.
        """
        self.current_task = task
        # Tìm environment dựa trên task
        self.environment = self.env_matcher.find_environment(task)
        
        return (
            f"I have received the task. Now, please provide the specific steps you want me to perform." )

    def set_planned_steps(self, steps, source="without_env"):
        """
        Lưu danh sách step do planner sinh để chạy tự động.
        """
        self.plan_source = source
        self.planned_steps = []
        for idx, step in enumerate(steps or [], 1):
            if isinstance(step, dict):
                action = str(step.get("action", "")).strip()
                target = str(step.get("target_object", "")).strip()
                step_id = step.get("step_id", idx)
                if action and action != "plan_step":
                    text = f"{action} {target}".strip()
                else:
                    text = target
                if text:
                    self.planned_steps.append({"step_id": step_id, "text": text})
            elif isinstance(step, str) and step.strip():
                self.planned_steps.append({"step_id": idx, "text": step.strip()})

    def get_planned_steps(self):
        return list(self.planned_steps)

    def handle_step(self, step_query, environment=None, history=None):
        """
        Giai đoạn 2: Xử lý từng bước (query). Với query thì thực hiện extract.
        Tìm top 5 vật thể phù hợp nhất từ environment và classify ambiguity.
        """
        # Extract entities (actions + objects) từ step query
        self.entities = self.extractor.extract(step_query)
        
        # Sử dụng environment từ start_task
        env = environment if environment is not None else self.environment
        
        if not env:
            print("[Warning] Không có environment để tìm vật thể")
            return {
                "entities": self.entities,
                "top_objects": [],
                "classification": None
            }
        
        # Chỉ dùng objects để embedding / matching
        if isinstance(self.entities, dict):
            extracted_objects = self.entities.get("objects", [])
        else:
            extracted_objects = self.entities
        
        top_objects = self.embedding_selector.select_top_objects(
            extracted_objects, 
            env, 
            top_k=3 
        )
        
        # Classify ambiguity nếu có classifier
        classification = None
        if self.classifier:
            classification = self.classifier.classify(
                self.current_task,
                step_query,
                self.entities,
                top_objects,
                history=history,
            )
        
        return {
            "entities": self.entities,
            "top_objects": top_objects,
            "classification": classification
        }

    def clarify_step(self, clarification, pending_step, pending_entities, pending_top_objects, history=None):
        """Re-classify với top_k gốc — skip entity extraction và embedding."""
        step_query = f"{pending_step} [User specified: {clarification}]"
        classification = None
        if self.classifier:
            classification = self.classifier.classify(
                self.current_task,
                step_query,
                pending_entities,
                pending_top_objects,
                history=history,
                is_clarification=True,
            )
        return {
            "entities": pending_entities,
            "top_objects": pending_top_objects,
            "classification": classification,
        }


class ResponseGenerator:
    def __init__(self, llm_instance, prompt_path=None):
        self.llm = llm_instance
        if prompt_path is None:
            prompt_path = Path(__file__).parent / "prompts" / "response.txt"
        try:
            with open(prompt_path, 'r', encoding='utf-8') as f:
                self.prompt_template = f.read()
        except Exception:
            self.prompt_template = None
    
    def generate(self, query, classification_result, action="", history=None):
        status = classification_result.get("status", "Unambiguous")
        viable = classification_result.get("viable_objects", [])
        label = classification_result.get("label", "None")

        if not self.prompt_template:
            return self._fallback(query, status, viable)

        history_text = "(none)"
        if history:
            lines = []
            for turn in history[-3:]:
                lines.append(f"User: {turn['user_message']}")
                lines.append(f"Robot: {turn['robot_message']}")
            history_text = "\n".join(lines)

        prompt = self.prompt_template
        prompt = prompt.replace("{query}", query)
        prompt = prompt.replace("{classification}", status)
        prompt = prompt.replace("{ambiguity_type}", label)
        prompt = prompt.replace("{viable_objects}", json.dumps(viable))
        prompt = prompt.replace("{action}", action)
        prompt = prompt.replace("{history}", history_text)

        try:
            return self.llm.generate(prompt).strip()
        except Exception:
            return self._fallback(query, status, viable)
    
    def _fallback(self, query, status, viable):
        if status == "Unambiguous":
            return f"Done — {query}. What's next?"
        return f"I see {len(viable)} options for that: {', '.join(viable)}. Which one should I use?"



if __name__ == "__main__":
    # Model generate: dùng cho extract entities và classify ambiguity
    dataset_path = Path(__file__).resolve().parent.parent / "ambik_dataset" / "ambik_test_900.csv"

    llm = LLM("ollama:qwen2.5:32b", {"max_new_tokens": 150, "temperature": 0.7})
    embed_model = LLM("ollama:mxbai-embed-large:latest", {})
    
    # Trích xuất entity từ step task current và tìm environment phù hợp của task đó
    extractor = EntityExtractor(llm)
    env_matcher = EnvironmentMatcher(dataset_path)
    embedding_selector = EmbeddingSelector(embed_model)
    classifier = AmbiguityClassifier(llm)
    responder = ResponseGenerator(llm)
    handler = TaskHandler(extractor, env_matcher, embedding_selector, classifier)
    planner = TaskPlanner(llm, dataset_path=dataset_path)

    session_store = SessionStore()

    GREETING = 'hi, robot kitchen'
    FAREWELL = 'thank you, robot kitchen'

    print(f'Say "{GREETING}" to start a session.')
    while True:
        opening = input("> ").strip().lower()
        if opening == GREETING:
            print("Hello! I'm ready. What task would you like me to do?")
            break
        else:
            print(f'Please say "{GREETING}" to begin.')

    session_id = session_store.new_session_id()
    print(f"[Session started: {session_id}]")

    task = input("Enter ambiguous task (no environment needed): ").strip()
    print(handler.start_task(task))

    plan_result = planner.plan(task)
    planned_steps = plan_result.get("steps_without_env", [])
    handler.set_planned_steps(planned_steps, source="without_env")

    if handler.get_planned_steps():
        print(f"\nGenerated plan WITHOUT environment ({len(handler.get_planned_steps())} steps):")
        for step in handler.get_planned_steps():
            print(f"  {step['step_id']}. {step['text']}")
    else:
        print("\n[Warning] Planner did not return steps. Nothing to execute.")

    session_store.save_session({
        "session_id": session_id,
        "task": task,
        "environment": handler.environment,
        "history": [],
    })

    conversation_history = []

    print(f'\n(Auto mode runs generated steps. Say "{FAREWELL}" when asked for clarification to end.)')

    # Chạy tuần tự các step planner sinh ra (without environment)
    for planned in handler.get_planned_steps():
        query = planned["text"]
        result_query = query
        print(f"\n[Planned Step {planned['step_id']}] {query}")

        result = handler.handle_step(query, history=conversation_history)

        entities = result.get("entities", {})
        if isinstance(entities, dict):
            actions = entities.get("actions", [])
            objects = entities.get("objects", [])
        else:
            actions = []
            objects = entities

        print(f"\nActions extracted: {actions}")
        print(f"Objects extracted: {objects}")

        print("Top objects:")
        for i, (obj, score) in enumerate(result['top_objects'], 1):
            print(f"  {i}. {obj:<30} (score: {score:.4f})")

        robot_response = ""
        if result.get('classification'):
            cls = result['classification']
            print(f"\nClassification: {cls['status']}")
            print(f"  Label: {cls['label']}")

            viable = cls.get('viable_objects', [])
            if viable:
                print(f"  Viable Objects: {viable}")

            action_str = ", ".join(actions) if actions else ""
            robot_response = responder.generate(query, cls, action_str, history=conversation_history)
            print(f"\n Robot: {robot_response}")

            # Nếu mơ hồ thì hỏi lại user, giữ đúng logic clarify hiện có
            while cls.get("status") == "Ambiguous":
                pending_step = result_query
                pending_entities = entities
                pending_top_objects = result.get("top_objects", [])
                clarification = input("Clarification needed. Please specify your choice: ").strip()
                if clarification.lower() == FAREWELL:
                    session_store.delete_session(session_id)
                    print("You're welcome! Goodbye.")
                    sys.exit(0)
                if not clarification:
                    print("Please provide a clarification.")
                    continue

                result = handler.clarify_step(
                    clarification,
                    pending_step,
                    pending_entities,
                    pending_top_objects,
                    history=conversation_history,
                )
                cls = result.get("classification", {})
                result_query = f"{pending_step} [User specified: {clarification}]"
                action_str = ", ".join(actions) if actions else ""
                robot_response = responder.generate(
                    result_query,
                    cls,
                    action_str,
                    history=conversation_history,
                )
                print(f"\n Robot (after clarification): {robot_response}")
                if cls.get("status") == "Ambiguous":
                    print("Still ambiguous, please clarify once more.")

        conversation_history.append({"user_message": result_query, "robot_message": robot_response})
        conversation_history = conversation_history[-3:]
        session_store.add_turn(session_id, task, handler.environment, result_query, robot_response)

    print("\nAll planned steps processed.")
    session_store.delete_session(session_id)
    print("You're welcome! Goodbye.")

