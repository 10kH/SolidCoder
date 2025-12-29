from typing import List, Dict
import os
import time

from tenacity import retry, stop_after_attempt, wait_random_exponential

from .Base import BaseModel

# HuggingFace token
HF_TOKEN = os.getenv("HF_TOKEN", "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")

# Model name mapping
MODEL_MAPPING = {
    "gemma-2-9b": "google/gemma-2-9b-it",
    "mixtral-8x7b": "mistralai/Mixtral-8x7B-Instruct-v0.1",
    "llama-3.1-8b": "meta-llama/Llama-3.1-8B-Instruct",
    "llama-3.1-70b": "meta-llama/Llama-3.1-70B-Instruct",
}


class HuggingFaceModel(BaseModel):
    """HuggingFace local model wrapper using transformers pipeline"""

    def __init__(self, model_name, sleep_time=0, **kwargs):
        if model_name is None:
            raise Exception("Model name is required")

        self.model_name = model_name
        self.hf_model_id = MODEL_MAPPING.get(model_name, model_name)
        self.temperature = kwargs.get("temperature", 0.0)
        self.top_p = kwargs.get("top_p", 0.95)
        self.max_tokens = kwargs.get("max_tokens", 16000)
        self.sleep_time = sleep_time

        # Lazy loading - don't load model in __init__
        self._pipeline = None
        self._tokenizer = None

    def _load_model(self):
        """Lazy load model and tokenizer"""
        if self._pipeline is not None:
            return

        import torch
        from transformers import pipeline, AutoTokenizer

        print(f"Loading model: {self.hf_model_id}")

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.hf_model_id,
            token=HF_TOKEN
        )

        self._pipeline = pipeline(
            "text-generation",
            model=self.hf_model_id,
            model_kwargs={"torch_dtype": torch.bfloat16},
            device_map="auto",
            token=HF_TOKEN
        )

        print(f"Model loaded: {self.hf_model_id}")

    @retry(wait=wait_random_exponential(min=10, max=60), stop=stop_after_attempt(3))
    def prompt(
        self,
        processed_input: List[Dict],
        frequency_penalty=0.0,
        presence_penalty=0.0,
    ) -> tuple[str, dict]:

        time.sleep(self.sleep_time)

        # Lazy load model
        self._load_model()

        start_time = time.perf_counter()

        # Convert messages to HF format
        messages = []
        for msg in processed_input:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            messages.append({"role": role, "content": content})

        # Generate
        do_sample = self.temperature > 0
        outputs = self._pipeline(
            messages,
            max_new_tokens=self.max_tokens,
            do_sample=do_sample,
            temperature=self.temperature if do_sample else None,
            top_p=self.top_p if do_sample else None,
            pad_token_id=self._tokenizer.eos_token_id,
        )

        # Extract generated text
        response_text = outputs[0]["generated_text"][-1]["content"]

        end_time = time.perf_counter()

        # Estimate tokens (rough approximation)
        prompt_text = " ".join([m["content"] for m in messages])
        prompt_tokens = len(self._tokenizer.encode(prompt_text))
        completion_tokens = len(self._tokenizer.encode(response_text))

        run_details = {
            "api_calls": 1,
            "taken_time": end_time - start_time,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "cost": 0,
            "details": [
                {
                    "model_name": self.model_name,
                    "model_prompt": processed_input,
                    "model_response": response_text,
                    "max_tokens": self.max_tokens,
                    "temperature": self.temperature,
                    "top_p": self.top_p,
                    "frequency_penalty": frequency_penalty,
                    "presence_penalty": presence_penalty
                }
            ],
        }

        return response_text, run_details
