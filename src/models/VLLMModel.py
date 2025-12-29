from typing import List, Dict, Optional
import os
import re
import time

from tenacity import retry, stop_after_attempt, wait_random_exponential

from .Base import BaseModel

# HuggingFace token for gated models
HF_TOKEN = os.getenv("HF_TOKEN", "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")

# Model name mapping
MODEL_MAPPING = {
    # Standard instruction-tuned models
    "gemma-2-9b": "google/gemma-2-9b-it",
    "mixtral-8x7b": "mistralai/Mixtral-8x7B-Instruct-v0.1",
    "llama-3.1-8b": "meta-llama/Llama-3.1-8B-Instruct",
    "llama-3.1-70b": "meta-llama/Llama-3.1-70B-Instruct",
    # Reasoning models (3B active → 7B → 8B → 20B)
    "qwen3-30b-a3b-thinking": "Qwen/Qwen3-30B-A3B-Thinking-2507",             # 30B MoE (3.3B active) - Qwen (AIME25: 85.0%)
    "openthinker3-7b": "open-thoughts/OpenThinker3-7B",                       # 7B - OpenThoughts (AIME24: 69.0%, AIME25: 53.3%)
    "deepseek-r1-0528-8b": "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B",           # 8B - DeepSeek (AIME24: 86.0%)
    "gpt-oss-20b": "openai/gpt-oss-20b",                                      # 20B MoE (3.6B active) - OpenAI (Apache 2.0)
}

# Tensor parallel size for each model (based on model size and GPU memory)
TENSOR_PARALLEL_SIZE = {
    # Standard models
    "gemma-2-9b": 1,      # ~20GB, fits on 1 A100
    "mixtral-8x7b": 2,    # ~90GB with MoE, needs 2 GPUs
    "llama-3.1-8b": 1,    # ~16GB, fits on 1 A100
    "llama-3.1-70b": 4,   # ~140GB, needs 4 GPUs
    # Reasoning models (all fit on 1 GPU with 32k context)
    "qwen3-30b-a3b-thinking": 1,   # 30B MoE (3.3B active), ~60GB
    "openthinker3-7b": 1,          # 7B, ~16GB (Qwen2.5-7B based)
    "deepseek-r1-0528-8b": 1,      # 8B, ~18GB
    "gpt-oss-20b": 1,              # 20B MoE (MXFP4), ~16GB
}

# Max model length for each model (based on model's max_position_embeddings)
MAX_MODEL_LEN = {
    # Standard models
    "gemma-2-9b": 8192,       # Gemma 2 has 8k context
    "mixtral-8x7b": 32768,    # Mixtral has 32k context
    "llama-3.1-8b": 32768,    # LLaMA 3.1 has 128k but we use 32k for efficiency
    "llama-3.1-70b": 32768,   # LLaMA 3.1 has 128k but we use 32k for efficiency
    # Reasoning models (use 32k for efficiency despite larger native context)
    "qwen3-30b-a3b-thinking": 32768,   # Native 262k, use 32k
    "openthinker3-7b": 32768,          # Native 128k, use 32k (Qwen2.5-7B based)
    "deepseek-r1-0528-8b": 32768,      # Native 64k, use 32k
    "gpt-oss-20b": 32768,              # Native 256k, use 32k
}

# Reasoning models with chain-of-thought capabilities
REASONING_MODELS = {
    "qwen3-30b-a3b-thinking",   # 30B MoE - Qwen (uses <think> tags, thinking-only mode)
    "openthinker3-7b",          # 7B - OpenThoughts (uses <think> tags, QwQ-style)
    "deepseek-r1-0528-8b",      # 8B - DeepSeek (uses <think> tags)
    "gpt-oss-20b",              # 20B - OpenAI (uses harmony format with reasoning levels)
}

# Models where chat template auto-adds thinking tag (no forced start needed)
AUTO_THINK_MODELS = {
    "qwen3-30b-a3b-thinking",   # Chat template auto-adds <think> (thinking-only mode)
    "openthinker3-7b",          # QwQ-style reasoning, auto-generates thinking
    "deepseek-r1-0528-8b",      # R1-0528: "It is not required to add <think> at the beginning"
    "gpt-oss-20b",              # Uses harmony format, auto-generates reasoning
}

# System prompts for reasoning models
REASONING_SYSTEM_PROMPTS = {
    "qwen3-30b-a3b-thinking": None,  # Thinking-only mode, no system prompt needed
    "openthinker3-7b": None,           # QwQ-style, no system prompt needed
    "deepseek-r1-0528-8b": None,       # System prompt supported but not required
    "gpt-oss-20b": "Reasoning: high",  # OpenAI harmony format - high reasoning effort
}

# Recommended sampling parameters for reasoning models
REASONING_SAMPLING_PARAMS = {
    "qwen3-30b-a3b-thinking": {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "min_p": 0},  # 30B MoE - Qwen recommended
    "openthinker3-7b": {"temperature": 0.7, "top_p": 0.95},                      # 7B - OpenThoughts (SFT, standard sampling)
    "deepseek-r1-0528-8b": {"temperature": 0.6, "top_p": 0.95},                  # 8B - DeepSeek recommended
    "gpt-oss-20b": {"temperature": 0.6, "top_p": 0.95},                          # 20B - OpenAI (similar to other reasoning models)
}


class VLLMModel(BaseModel):
    """vLLM-based local model wrapper for fast inference"""

    def __init__(self, model_name, sleep_time=0, **kwargs):
        if model_name is None:
            raise Exception("Model name is required")

        self.model_name = model_name
        self.hf_model_id = MODEL_MAPPING.get(model_name, model_name)
        self.sleep_time = sleep_time

        # Check if this is a reasoning model
        self.is_reasoning_model = model_name in REASONING_MODELS

        # For reasoning models, use recommended sampling params unless overridden
        if self.is_reasoning_model and model_name in REASONING_SAMPLING_PARAMS:
            default_params = REASONING_SAMPLING_PARAMS[model_name]
            self.temperature = kwargs.get("temperature", default_params.get("temperature", 0.6))
            self.top_p = kwargs.get("top_p", default_params.get("top_p", 0.95))
            self.top_k = kwargs.get("top_k", default_params.get("top_k", 40))
        else:
            self.temperature = kwargs.get("temperature", 0.0)
            self.top_p = kwargs.get("top_p", 0.95)
            self.top_k = kwargs.get("top_k", -1)  # -1 means disabled

        self.max_tokens = kwargs.get("max_tokens", 16000)

        # Whether to extract only the solution part (after </think>) for reasoning models
        # Set to False to keep full response including thinking
        self.extract_solution_only = kwargs.get("extract_solution_only", True)

        # Whether to keep thinking content in run_details for analysis
        self.save_thinking = kwargs.get("save_thinking", True)

        # Get tensor parallel size from mapping or kwargs
        self.tensor_parallel_size = kwargs.get(
            "tensor_parallel_size",
            TENSOR_PARALLEL_SIZE.get(model_name, 1)
        )

        # Get max model length from mapping or kwargs
        self.max_model_len = kwargs.get(
            "max_model_len",
            MAX_MODEL_LEN.get(model_name, 32768)
        )

        # Cap max_tokens at max_model_len to prevent errors
        if self.max_tokens > self.max_model_len:
            print(f"Warning: max_tokens ({self.max_tokens}) > max_model_len ({self.max_model_len}), capping to {self.max_model_len - 1024}")
            self.max_tokens = self.max_model_len - 1024  # Leave room for prompt

        # Lazy loading - don't load model in __init__
        self._llm = None
        self._tokenizer = None

        if self.is_reasoning_model:
            print(f"Reasoning model detected: {model_name}")
            print(f"  - Temperature: {self.temperature}, Top-P: {self.top_p}, Top-K: {self.top_k}")
            print(f"  - Extract solution only: {self.extract_solution_only}")

    def _load_model(self):
        """Lazy load vLLM model"""
        if self._llm is not None:
            return

        from vllm import LLM

        print(f"Loading vLLM model: {self.hf_model_id} (tensor_parallel_size={self.tensor_parallel_size}, max_model_len={self.max_model_len})")

        self._llm = LLM(
            model=self.hf_model_id,
            tensor_parallel_size=self.tensor_parallel_size,
            trust_remote_code=True,
            dtype="bfloat16",
            max_model_len=self.max_model_len,
            gpu_memory_utilization=0.90,  # Use 90% of GPU memory
        )

        self._tokenizer = self._llm.get_tokenizer()

        print(f"vLLM model loaded: {self.hf_model_id}")

    def _extract_thinking_and_solution(self, response_text: str) -> tuple[str, str]:
        """
        Extract thinking and solution parts from reasoning model output.
        All current models use <think>...</think> format.

        Returns:
            tuple: (thinking_content, solution_content)
        """
        # Match <think>...</think> pattern (Qwen, DeepSeek, Phi-4)
        think_pattern = r'<think>(.*?)</think>'
        match = re.search(think_pattern, response_text, re.DOTALL)

        if match:
            thinking = match.group(1).strip()
            solution = response_text[match.end():].strip()
            return thinking, solution

        # No thinking block found, return entire response as solution
        return "", response_text

    def _prepare_messages_for_reasoning(self, messages: List[Dict]) -> List[Dict]:
        """
        Prepare messages for reasoning models by adding appropriate system prompt.
        """
        # Check if system prompt is needed for this model
        system_prompt = REASONING_SYSTEM_PROMPTS.get(self.model_name)

        if system_prompt:
            # Check if there's already a system message
            has_system = any(msg.get("role") == "system" for msg in messages)
            if not has_system:
                # Prepend system message
                return [{"role": "system", "content": system_prompt}] + messages

        return messages

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

        from vllm import SamplingParams

        start_time = time.perf_counter()

        # Convert messages to chat format
        messages = []
        for msg in processed_input:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            messages.append({"role": role, "content": content})

        # For reasoning models, add system prompt if needed
        if self.is_reasoning_model:
            messages = self._prepare_messages_for_reasoning(messages)

        # Apply chat template to get the prompt
        prompt_text = self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        # Most reasoning models auto-add thinking tags via chat template
        # Only force <think> for models that need it (currently none in our list)
        if self.is_reasoning_model and self.model_name not in AUTO_THINK_MODELS and self.model_name != "phi-4-reasoning":
            # Legacy fallback for models that need forced <think>
            prompt_text = prompt_text + "<think>\n"

        # Set up sampling parameters
        sampling_kwargs = {
            "temperature": self.temperature if self.temperature > 0 else 0.0,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
            "frequency_penalty": frequency_penalty,
            "presence_penalty": presence_penalty,
        }

        # Add top_k for reasoning models if set
        if self.top_k > 0:
            sampling_kwargs["top_k"] = self.top_k

        sampling_params = SamplingParams(**sampling_kwargs)

        # Generate
        outputs = self._llm.generate([prompt_text], sampling_params)

        # Extract generated text
        raw_response = outputs[0].outputs[0].text

        # For reasoning models that we forced to start with <think>, prepend it back
        if self.is_reasoning_model and self.model_name not in AUTO_THINK_MODELS and self.model_name != "phi-4-reasoning":
            raw_response = "<think>\n" + raw_response

        end_time = time.perf_counter()

        # Get token counts from vLLM output
        prompt_tokens = len(outputs[0].prompt_token_ids)
        completion_tokens = len(outputs[0].outputs[0].token_ids)

        # Process response for reasoning models
        thinking_content = ""
        response_text = raw_response

        if self.is_reasoning_model:
            thinking_content, solution_content = self._extract_thinking_and_solution(raw_response)
            if self.extract_solution_only and solution_content:
                response_text = solution_content
            else:
                response_text = raw_response

        run_details = {
            "api_calls": 1,
            "taken_time": end_time - start_time,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "cost": 0,  # Local model, no cost
            "details": [
                {
                    "model_name": self.model_name,
                    "model_prompt": processed_input,
                    "model_response": response_text,
                    "raw_response": raw_response if self.is_reasoning_model else None,
                    "thinking_content": thinking_content if self.save_thinking else None,
                    "max_tokens": self.max_tokens,
                    "temperature": self.temperature,
                    "top_p": self.top_p,
                    "top_k": self.top_k if self.top_k > 0 else None,
                    "frequency_penalty": frequency_penalty,
                    "presence_penalty": presence_penalty,
                    "is_reasoning_model": self.is_reasoning_model,
                }
            ],
        }

        return response_text, run_details


# Utility functions for model information
def list_available_models():
    """List all available models with their configurations."""
    print("\n" + "=" * 80)
    print("Available vLLM Models")
    print("=" * 80)

    print("\n[Standard Instruction-Tuned Models]")
    print("-" * 60)
    print(f"{'Model Name':<25} {'HF Model ID':<40} {'GPUs':<5} {'Context':<8}")
    print("-" * 60)
    for name in ["gemma-2-9b", "mixtral-8x7b", "llama-3.1-8b", "llama-3.1-70b"]:
        if name in MODEL_MAPPING:
            print(f"{name:<25} {MODEL_MAPPING[name]:<40} {TENSOR_PARALLEL_SIZE.get(name, 1):<5} {MAX_MODEL_LEN.get(name, 32768):<8}")

    print("\n[Reasoning Models (with <think> support)]")
    print("-" * 60)
    print(f"{'Model Name':<25} {'HF Model ID':<40} {'GPUs':<5} {'Context':<8}")
    print("-" * 60)
    for name in REASONING_MODELS:
        if name in MODEL_MAPPING:
            print(f"{name:<25} {MODEL_MAPPING[name]:<40} {TENSOR_PARALLEL_SIZE.get(name, 1):<5} {MAX_MODEL_LEN.get(name, 32768):<8}")

    print("\n" + "=" * 80)


def get_model_gpu_requirements(model_name: str) -> dict:
    """
    Get GPU requirements for a specific model.

    Args:
        model_name: Name of the model

    Returns:
        dict with model configuration
    """
    if model_name not in MODEL_MAPPING:
        return {"error": f"Unknown model: {model_name}"}

    return {
        "model_name": model_name,
        "hf_model_id": MODEL_MAPPING[model_name],
        "tensor_parallel_size": TENSOR_PARALLEL_SIZE.get(model_name, 1),
        "max_model_len": MAX_MODEL_LEN.get(model_name, 32768),
        "is_reasoning_model": model_name in REASONING_MODELS,
        "recommended_params": REASONING_SAMPLING_PARAMS.get(model_name, {}),
    }


def print_reasoning_model_info():
    """Print detailed information about reasoning models."""
    print("\n" + "=" * 80)
    print("Reasoning Model Configuration Guide")
    print("=" * 80)

    # Order by size
    ordered_models = ["qwen3-4b-thinking", "openthinker3-7b", "deepseek-r1-0528-8b", "phi-4-reasoning"]

    for name in ordered_models:
        if name in MODEL_MAPPING:
            print(f"\n[{name}]")
            print(f"  HuggingFace ID: {MODEL_MAPPING[name]}")
            print(f"  GPUs Required:  {TENSOR_PARALLEL_SIZE.get(name, 1)} x A100 80GB")
            print(f"  Context Length: {MAX_MODEL_LEN.get(name, 32768)} tokens")

            params = REASONING_SAMPLING_PARAMS.get(name, {})
            if params:
                print(f"  Recommended Sampling:")
                print(f"    - temperature: {params.get('temperature', 0.6)}")
                print(f"    - top_p: {params.get('top_p', 0.95)}")
                if 'top_k' in params:
                    print(f"    - top_k: {params.get('top_k')}")

            print(f"  Thinking Format: <think>...</think>")

            sys_prompt = REASONING_SYSTEM_PROMPTS.get(name)
            if sys_prompt:
                print(f"  System Prompt:  Required (auto-added)")
            else:
                print(f"  System Prompt:  Not needed")

    print("\n" + "=" * 80)
    print("Usage Example:")
    print("-" * 80)
    print("""
# All reasoning models fit on 1 GPU - run 8 experiments in parallel!

# Qwen3 4B (AIME25: 81.3%):
CUDA_VISIBLE_DEVICES=0 python src/main.py \\
    --strategy CodeSIM \\
    --dataset HumanEval \\
    --model qwen3-4b-thinking \\
    --model_provider vLLM

# DeepSeek R1-0528 8B (AIME24: 86.0%):
CUDA_VISIBLE_DEVICES=1 python src/main.py \\
    --strategy CodeSIM \\
    --dataset HumanEval \\
    --model deepseek-r1-0528-8b \\
    --model_provider vLLM

# Ministral 14B (AIME25: 85.0%):
CUDA_VISIBLE_DEVICES=2 python src/main.py \\
    --strategy CodeSIM \\
    --dataset HumanEval \\
    --model ministral-14b-reasoning \\
    --model_provider vLLM

# Phi-4 Reasoning 14B (AIME24: 75.3%):
CUDA_VISIBLE_DEVICES=3 python src/main.py \\
    --strategy CodeSIM \\
    --dataset HumanEval \\
    --model phi-4-reasoning \\
    --model_provider vLLM
""")
    print("=" * 80)


if __name__ == "__main__":
    # When run directly, print model information
    list_available_models()
    print_reasoning_model_info()
