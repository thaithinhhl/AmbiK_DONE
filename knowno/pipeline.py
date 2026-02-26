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
    # Model generate: dùng cho extract entities và classify ambiguity
    dataset_path = Path(__file__).resolve().parent.parent / "ambik_dataset" / "ambik_test_900.csv"

    llm = LLM("ollama:llama3.1:8b", {"max_new_tokens": 150, "temperature": 0.7})
    embed_model = LLM("ollama:mxbai-embed-large:latest", {})
    
    # Trích xuất entity từ step task current và tìm environment phù hợp của task đó
    extractor = EntityExtractor(llm)
    env_matcher = EnvironmentMatcher(dataset_path)
    
    # Thực hiện tìm top K vật thể phù hợp dựa vào task current và thông tin từ môi trường
    embedding_selector = EmbeddingSelector(embed_model)
    classifier = AmbiguityClassifier(llm)
    responder = ResponseGenerator(llm)
    
    handler = TaskHandler(extractor, env_matcher, embedding_selector, classifier)

    task = input("Enter task: ")
    print(handler.start_task(task))

    while True:
        query = input("\nEnter step query (hoặc 'quit' để kết thúc): ").strip()
        if query.lower() in ("quit", "exit", "q", ""):
            break
        
        result = handler.handle_step(query)
        
        entities = result.get("entities", {})
        if isinstance(entities, dict):
            actions = entities.get("actions", [])
            objects = entities.get("objects", [])
        else:
            actions = []
            objects = entities
        
        print(f"\nActions extracted: {actions}")
        print(f"Objects extracted: {objects}")
        
        # In top objects với score
        print(f"Top objects:")
        for i, (obj, score) in enumerate(result['top_objects'], 1):
            print(f"  {i}. {obj:<30} (score: {score:.4f})")
        
        # Hiển thị classification nếu có
        if result.get('classification'):
            cls = result['classification']
            print(f"\nClassification: {cls['status']}")
            print(f"  Label: {cls['label']}")
            
            viable = cls.get('viable_objects', [])
            if viable:
                print(f"  Viable Objects: {viable}")
            
            action_str = ", ".join(actions) if actions else ""
            response = responder.generate(query, cls, action_str)
            print(f"\n🤖 Robot: {response}")
            
