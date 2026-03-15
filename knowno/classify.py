import json
import re


class AmbiguityClassifier:
    """
    Phân loại step query là Unambiguous hoặc Ambiguous (Preferences, Common Sense, Safety)
    dựa trên task, entities, và top-K objects.
    """
    
    def __init__(self, llm_instance, prompt_path=None):
        """
        :param llm_instance: Instance của LLM
        :param prompt_path: Đường dẫn đến file prompt template (mặc định: prompts/classify_amb.txt)
        """
        self.llm = llm_instance
        self.prompt_template = self.load_prompt(prompt_path)
    
    def load_prompt(self, prompt_path=None):
        if prompt_path is None:
            from pathlib import Path
            prompt_path = Path(__file__).parent / "prompts" / "classify_amb.txt"
        
        try:
            with open(prompt_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            print(f"[Warning] Không thể đọc prompt từ {prompt_path}: {e}")
            return None
    
    def classify(self, task, step_query, entities, top_objects, history=None):
        if top_objects and isinstance(top_objects[0], tuple):
            top_k_list = [obj for obj, *_ in top_objects]
        else:
            top_k_list = list(top_objects)
        
        if isinstance(entities, dict):
            actions = entities.get("actions", [])
            objects = entities.get("objects", [])
        else:
            actions = []
            objects = []
        
        action_str = ", ".join(actions) if actions else ""
        object_str = json.dumps(objects) if objects else "[]"

        # Xây dựng history text từ conversation history (tối đa 3 lượt gần nhất)
        history_text = "(none)"
        if history:
            lines = []
            for turn in history[-3:]:
                lines.append(f"User: {turn['user_message']}")
                lines.append(f"Robot: {turn['robot_message']}")
            history_text = "\n".join(lines)

        # Thay placeholder trong prompt template
        prompt = (self.prompt_template or "")
        prompt = prompt.replace("{history}", history_text)
        prompt = prompt.replace("{query}", step_query)
        prompt = prompt.replace("{action}", action_str)
        prompt = prompt.replace("{object}", object_str)
        prompt = prompt.replace("{top_k_env_objects}", json.dumps(top_k_list))
        
        # Gọi LLM
        response_text = self.llm.generate(prompt)
        
        # Parse JSON đơn giản
        result = None
        try:
            result = json.loads(response_text.strip())
        except:
            for match in re.finditer(r'\{[^{}]+\}', response_text):
                try:
                    result = json.loads(match.group())
                    if "classification" in result or "ambiguity_type" in result:
                        break
                except:
                    pass
        
        if result is None:
            print("[Warning] Không parse được JSON classification; mặc định Unambiguous.")
            return {
                "status": "Unambiguous",
                "label": "None",
                "viable_objects": [],
            }
        
        status = result.get("classification", result.get("status", "Unambiguous"))
        label = result.get("ambiguity_type", result.get("label", "None"))
        viable_objects = result.get("viable_objects", [])
        
        # VALIDATION: viable_objects phải là subset của top_k_list (tránh LLM hallucination)
        if viable_objects and isinstance(viable_objects, list):
            top_k_set = set(obj.lower().strip() for obj in top_k_list)
            validated_viable = [
                obj for obj in viable_objects
                if obj and obj.lower().strip() in top_k_set
            ]
            if len(validated_viable) < len(viable_objects):
                print("[Warning] LLM hallucination: viable_objects không nằm trong Top K.")
            viable_objects = validated_viable
            
        if len(viable_objects) >= 2:
            status = "Ambiguous"
        else:
            status = "Unambiguous"
            label = "None"

        return {
            "status": status,
            "label": label,
            "viable_objects": viable_objects if isinstance(viable_objects, list) else [],
        }


if __name__ == "__main__":
    import sys
    from pathlib import Path
    
    _root = Path(__file__).resolve().parent.parent
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))
    
    from llm import LLM
    
    # Test
    model = LLM("ollama:qwen2.5:32b", {"max_new_tokens": 150, "temperature": 0.7})
    classifier = AmbiguityClassifier(model)
    