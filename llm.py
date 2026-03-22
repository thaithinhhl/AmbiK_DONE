import numpy as np
import requests

# Ollama API configuration
OLLAMA_BASE_URL = "http://localhost:11434"  # Default Ollama API endpoint


def temperature_scaling(logits, temperature=1):
    logits = np.array(logits)
    logits /= temperature
    try:
        logits -= logits.max()
    except Exception:
        pass
    logits = logits - np.log(np.sum(np.exp(logits)))
    smx = np.exp(logits)
    return [float(x) for x in smx]


class LLM:
    """Wrapper chỉ hỗ trợ Ollama API."""

    def __init__(self, model_name, generation_settings):
        if not model_name.startswith("ollama:"):
            raise ValueError("Chỉ hỗ trợ Ollama. Model name phải có dạng 'ollama:<tên_model>'")
        self.model_name = model_name
        self.ollama_model = model_name.replace("ollama:", "")
        self.ollama_base_url = OLLAMA_BASE_URL
        self.generation_settings = generation_settings

    def filter_logits(self, logits, words, use_softmax=True):
        """Lọc logits từ response Ollama (dict) theo từng phương án."""
        filtered_logits = []
        filtered_letters = []
        if isinstance(logits, dict):
            for word in words:
                word_upper = word.upper()
                if word_upper in logits:
                    filtered_logits.append(logits[word_upper])
                    filtered_letters.append(word_upper)
                elif word in logits:
                    filtered_logits.append(logits[word])
                    filtered_letters.append(word)
        if use_softmax and len(filtered_logits) > 0:
            filtered_logits = temperature_scaling(filtered_logits)
        return dict(zip(filtered_letters, filtered_logits))

    def _parse_logprobs(self, result, debug=False):
        """Trích xuất logprobs cho A/B/C/D từ response Ollama (cần Ollama v0.12.11+)."""
        logprobs_data = result.get("logprobs") or result.get("eval")
        if not logprobs_data or len(logprobs_data) == 0:
            if debug:
                keys = list(result.keys())
                print(f"[Ollama] Response keys: {keys}. Không có 'logprobs'/'eval'.")
                print(f"[Ollama] Cần nâng cấp lên v0.12.11+ để có logprobs. Hiện tại: ollama --version")
            return {}
        # Lấy token đầu tiên (câu trả lời A/B/C/D)
        first_eval = logprobs_data[0]
        top_logprobs = first_eval.get("top_logprobs")
        if not top_logprobs:
            # Token được chọn
            token = first_eval.get("token", "").strip().upper()
            logprob = first_eval.get("logprob", -999)
            return {token: logprob} if token in "ABCD1234" else {}
        # Build dict: token -> logprob, chuẩn hóa A/B/C/D
        mapping = {"1": "A", "2": "B", "3": "C", "4": "D"}
        logits_dict = {}
        for item in top_logprobs:
            t = (item.get("token") or "").strip()
            if not t:
                continue
            lp = item.get("logprob", -999)
            key = t.upper() if t.upper() in "ABCD" else mapping.get(t, t)
            logits_dict[key] = lp
            if t in "1234":
                logits_dict[mapping[t]] = lp
        return logits_dict

    def _call_ollama_api(self, prompt, return_logits=False):
        """Gọi Ollama API để sinh văn bản."""
        url = f"{self.ollama_base_url}/api/generate"
        ollama_params = {
            "model": self.ollama_model,
            "prompt": prompt,
            "stream": False,
        }
        if "max_new_tokens" in self.generation_settings:
            ollama_params["num_predict"] = self.generation_settings["max_new_tokens"]
        elif "max_tokens" in self.generation_settings:
            ollama_params["num_predict"] = self.generation_settings["max_tokens"]
        if "temperature" in self.generation_settings:
            ollama_params["temperature"] = self.generation_settings["temperature"]
        if "top_p" in self.generation_settings:
            ollama_params["top_p"] = self.generation_settings["top_p"]
        if "stop" in self.generation_settings and self.generation_settings["stop"]:
            ollama_params["stop"] = self.generation_settings["stop"]
        if return_logits:
            ollama_params["options"] = ollama_params.get("options", {})
            ollama_params["options"]["logprobs"] = True
            ollama_params["options"]["top_logprobs"] = 20

        try:
            response = requests.post(url, json=ollama_params, timeout=120)
            response.raise_for_status()
            result = response.json()
            generated_text = result.get("response", "").strip()
            if return_logits:
                first_empty = not getattr(self, "_logprobs_warned", False)
                logits = self._parse_logprobs(result, debug=first_empty)
                if not logits:
                    self._logprobs_warned = True
                return (generated_text, [[logits]])
            return generated_text
        except requests.exceptions.RequestException as e:
            raise Exception(f"Ollama API error: {e}")

    def chat(self, system_prompt, user_prompt, return_logits=False):
        """Gọi Ollama Chat API với system/user role tách biệt (ChatML)."""
        url = f"{self.ollama_base_url}/api/chat"
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})

        options = {}
        if "max_new_tokens" in self.generation_settings:
            options["num_predict"] = self.generation_settings["max_new_tokens"]
        elif "max_tokens" in self.generation_settings:
            options["num_predict"] = self.generation_settings["max_tokens"]
        if "temperature" in self.generation_settings:
            options["temperature"] = self.generation_settings["temperature"]
        if "top_p" in self.generation_settings:
            options["top_p"] = self.generation_settings["top_p"]
        if return_logits:
            options["logprobs"] = True
            options["top_logprobs"] = 20

        ollama_params = {
            "model": self.ollama_model,
            "messages": messages,
            "stream": False,
            "options": options,
        }
        if "stop" in self.generation_settings and self.generation_settings["stop"]:
            ollama_params["stop"] = self.generation_settings["stop"]

        try:
            response = requests.post(url, json=ollama_params, timeout=120)
            response.raise_for_status()
            result = response.json()
            generated_text = result.get("message", {}).get("content", "").strip()
            if return_logits:
                first_empty = not getattr(self, "_logprobs_warned", False)
                logits = self._parse_logprobs(result, debug=first_empty)
                if not logits:
                    self._logprobs_warned = True
                return (generated_text, [[logits]])
            return generated_text
        except requests.exceptions.RequestException as e:
            raise Exception(f"Ollama Chat API error: {e}")

    def generate(self, prompt, return_logits=False):
        return self._call_ollama_api(prompt, return_logits=return_logits)

    def generate_batch(self, prompts, return_logits=False):
        """Ollama không hỗ trợ batch, gọi tuần tự."""
        generated_texts = []
        filtered_logits_li = []
        for prompt in prompts:
            result = self._call_ollama_api(prompt, return_logits=return_logits)
            if return_logits:
                generated_text, logits = result
                generated_texts.append(generated_text)
                filtered_logits_li.append(logits)
            else:
                generated_texts.append(result)
        if return_logits:
            return (generated_texts, filtered_logits_li)
        return generated_texts

    def embed(self, text):
        """
        Lấy embedding vector từ Ollama API.
        
        :param text: Text cần embedding
        :return: numpy array của embedding vector
        """
        url = f"{self.ollama_base_url}/api/embeddings"
        payload = {
            "model": self.ollama_model,
            "prompt": text
        }
        
        try:
            response = requests.post(url, json=payload, timeout=120)
            response.raise_for_status()
            result = response.json()
            embedding = result.get("embedding", [])
            if not embedding:
                raise ValueError(f"Ollama API không trả về embedding cho text: {text}")
            return np.array(embedding)
        except requests.exceptions.RequestException as e:
            raise Exception(f"Ollama Embedding API error: {e}")


if __name__ == "__main__":
    model = LLM("ollama:llama3.1:8b", {"max_new_tokens": 50, "temperature": 0.7})
    print(model.generate("Choose one letter A/B/C?"))
