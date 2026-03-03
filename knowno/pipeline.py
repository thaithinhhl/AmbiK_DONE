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
from knowno.history import ChatHistory

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

    def start_task(self, task):
        """
        Giai đoạn 1: Tiếp nhận task tổng quát (chỉ gọi 1 lần).
        Không thực hiện extract, chỉ lưu task và tìm environment tương ứng.
        """
        self.current_task = task
        # Tìm environment dựa trên task
        self.environment = self.env_matcher.find_environment(task)
        
        return (
            f"Tôi đã nhận nhiệm vụ. Bây giờ, hãy cung cấp các bước cụ thể bạn muốn tôi thực hiện." )

    def handle_step(self, step_query, environment=None):
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
                top_objects
            )
        
        return {
            "entities": self.entities,
            "top_objects": top_objects,
            "classification": classification
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
    
    def generate(self, query, classification_result, action=""):
        status = classification_result.get("status", "Unambiguous")
        viable = classification_result.get("viable_objects", [])
        label = classification_result.get("label", "None")
        
        # Fallback nếu không có prompt template
        if not self.prompt_template:
            return self._fallback(query, status, viable)
        
        prompt = self.prompt_template
        prompt = prompt.replace("{query}", query)
        prompt = prompt.replace("{classification}", status)
        prompt = prompt.replace("{ambiguity_type}", label)
        prompt = prompt.replace("{viable_objects}", json.dumps(viable))
        prompt = prompt.replace("{action}", action)
        
        try:
            return self.llm.generate(prompt).strip()
        except Exception:
            return self._fallback(query, status, viable)
    
    def _fallback(self, query, status, viable):
        if status == "Unambiguous":
            return f"Done — {query}. What's next?"
        return f"I see {len(viable)} options for that: {', '.join(viable)}. Which one should I use?"



if __name__ == "__main__":
    dataset_path = Path(__file__).resolve().parent.parent / "ambik_dataset" / "ambik_test_900.csv"

    llm = LLM("ollama:llama3.1:8b", {"max_new_tokens": 150, "temperature": 0.7})
    embed_model = LLM("ollama:mxbai-embed-large:latest", {})

    extractor = EntityExtractor(llm)
    env_matcher = EnvironmentMatcher(dataset_path)
    embedding_selector = EmbeddingSelector(embed_model)
    classifier = AmbiguityClassifier(llm)
    responder = ResponseGenerator(llm)
    handler = TaskHandler(extractor, env_matcher, embedding_selector, classifier)

    chat_history = ChatHistory(host="localhost", port=6379, db=0)

    START_SESSION = "hey, robot kitchen"
    END_SESSION = "thank you, robot kitchen"

    while True:
        user_input = input("You: ").strip()

        if user_input.lower() == "quit":
            if chat_history.current_session_id:
                sid, _ = chat_history.end_session()
                print(f"[System] Session {sid} closed.")
            print("[System] Exiting.")
            break

        # -- Start session --
        if user_input.lower() == START_SESSION:
            if chat_history.current_session_id:
                print(f"[System] Session {chat_history.current_session_id} is already open.")
                continue

            session_data = chat_history.start_session()
            sid = session_data["session_id"]
            print(f"[System] New session: {sid}")

            task = input("Enter task: ").strip()
            if not task:
                print("[System] No task provided.")
                continue

            greeting = handler.start_task(task)
            environment = handler.environment

            chat_history.set_task_context(task, environment)
            print(f"[System] Task + environment saved to Redis.")
            print(f"Robot: {greeting}")
            chat_history.add_message(task, greeting)
            continue

        # -- End session --
        if user_input.lower() == END_SESSION:
            if not chat_history.current_session_id:
                print(f'[System] No active session. Say "{START_SESSION}" to begin.')
                continue

            sid, history_data = chat_history.end_session()
            handler.current_task = None
            handler.environment = None
            handler.entities = None

            print(f"Robot: Thank you! Session {sid} ended.")
            if history_data:
                turns = len(history_data.get("history", []))
                print(f"[System] {turns} turns recorded.")
            continue

        # -- Step query --
        if not chat_history.current_session_id:
            print(f'[System] No active session. Say "{START_SESSION}" to begin.')
            continue

        stored_task, stored_env = chat_history.get_task_context()

        if not stored_task:
            print("[System] No task in session. Please enter a task.")
            task = input("Enter task: ").strip()
            if task:
                greeting = handler.start_task(task)
                chat_history.set_task_context(task, handler.environment)
                print(f"Robot: {greeting}")
                chat_history.add_message(task, greeting)
            continue

        handler.current_task = stored_task
        handler.environment = stored_env

        result = handler.handle_step(user_input, environment=stored_env)

        entities = result.get("entities", {})
        actions = entities.get("actions", []) if isinstance(entities, dict) else []
        objects = entities.get("objects", []) if isinstance(entities, dict) else entities

        print(f"\nActions: {actions}")
        print(f"Objects: {objects}")

        print("Top objects:")
        for i, (obj, score) in enumerate(result["top_objects"], 1):
            print(f"  {i}. {obj:<30} (score: {score:.4f})")

        robot_response = ""

        if result.get("classification"):
            cls = result["classification"]
            print(f"\nClassification: {cls['status']}")
            print(f"  Label: {cls['label']}")

            viable = cls.get("viable_objects", [])
            if viable:
                print(f"  Viable Objects: {viable}")

            action_str = ", ".join(actions) if actions else ""
            robot_response = responder.generate(user_input, cls, action_str)
        else:
            robot_response = "Step processed."

        print(f"\nRobot: {robot_response}")
        chat_history.add_message(user_input, robot_response)

