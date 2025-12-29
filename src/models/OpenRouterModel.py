import os
import time

from tenacity import retry, stop_after_attempt, wait_random_exponential
from openai import OpenAI

from .Base import BaseModel

usage_log_file_path = "openrouter_usage_log.csv"

# OpenRouter model name mappings
MODEL_MAPPINGS = {
    # OpenAI models
    "gpt-4o": "openai/gpt-4o",
    "gpt-4o-mini": "openai/gpt-4o-mini",
    "gpt-5-mini": "openai/gpt-5-mini",

    # Claude models
    "claude-haiku-4.5": "anthropic/claude-haiku-4.5",
    "claude-sonnet-4": "anthropic/claude-sonnet-4",

    # DeepSeek models
    "deepseek-v3.2-speciale": "deepseek/deepseek-v3.2-speciale",
    "deepseek-v3.2": "deepseek/deepseek-v3.2",
    "deepseek-r1-8b": "deepseek/deepseek-r1-0528-qwen3-8b",
    "deepseek-r1-llama70b": "deepseek/deepseek-r1-distill-llama-70b",

    # Gemini models
    "gemini-2.5-flash": "google/gemini-2.5-flash-preview-09-2025",
    "gemini-2.5-flash-preview-09-2025": "google/gemini-2.5-flash-preview-09-2025",
    "gemma-3-27b": "google/gemma-3-27b-it",

    # xAI Grok
    "grok-4.1-fast": "x-ai/grok-4.1-fast",
    "grok-3-mini": "x-ai/grok-3-mini",

    # Moonshot Kimi (paid - 0905 is the latest version with 256k context)
    "kimi-k2": "moonshotai/kimi-k2-0905",

    # Microsoft models
    "phi-4-reasoning": "microsoft/phi-4-reasoning-plus",

    # Mistral models
    "ministral-14b": "mistralai/ministral-14b-2512",

    # GLM models
    "glm-4.5-air": "z-ai/glm-4.5-air",
    "glm-4-32b": "z-ai/glm-4-32b",

    # OpenAI OSS models
    "gpt-oss-120b": "openai/gpt-oss-120b",

    # Meta Llama models
    "llama-4-maverick": "meta-llama/llama-4-maverick",
}


class OpenRouterModel(BaseModel):
    def __init__(
        self,
        model_name,
        sleep_time=0,
        **kwargs
    ):
        if model_name is None:
            raise Exception("Model name is required")

        # Map short names to full OpenRouter model names
        if model_name in MODEL_MAPPINGS:
            self.model_name = MODEL_MAPPINGS[model_name]
        elif "/" in model_name:
            # Already a full model name (e.g., "anthropic/claude-haiku-4.5")
            self.model_name = model_name
        else:
            raise Exception(f"Unknown model name: {model_name}. Available: {list(MODEL_MAPPINGS.keys())}")

        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv("OPENROUTER_API_KEY"),
        )

        self.temperature = kwargs.get("temperature", 0.0)
        self.top_p = kwargs.get("top_p", 0.95)
        self.max_tokens = kwargs.get("max_tokens", 8000)

        self.sleep_time = sleep_time

        print(f"OpenRouter model initialized: {self.model_name}")

    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(5))
    def prompt(
        self,
        processed_input: list[dict],
        frequency_penalty=0,
        presence_penalty=0
    ):
        time.sleep(self.sleep_time)

        start_time = time.perf_counter()

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": processed_input[0]["content"]
                        },
                    ]
                }
            ],
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_tokens,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
        )
        print(response.choices[0].message.content)

        end_time = time.perf_counter()

        with open(usage_log_file_path, mode="a") as file:
            file.write(f'{self.model_name},{response.usage.prompt_tokens},{response.usage.completion_tokens}\n')

        run_details = {
            "api_calls": 1,
            "taken_time": end_time - start_time,

            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,

            "details": [
                {
                    "model_name": self.model_name,
                    "model_prompt": processed_input,
                    "model_response": response.choices[0].message.content,
                    "max_tokens": self.max_tokens,
                    "temperature": self.temperature,
                    "top_p": self.top_p,
                    "frequency_penalty": frequency_penalty,
                    "presence_penalty": presence_penalty
                }
            ],
        }

        return response.choices[0].message.content, run_details
