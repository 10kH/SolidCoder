from models.Anthropic import *
from models.Gemini import *
from models.OpenAI import *
from models.GroqModel import *
from models.HuggingFaceModel import *
from models.VLLMModel import *
from models.OpenRouterModel import OpenRouterModel

class ModelFactory:
    @staticmethod
    def get_model_class(model_provider_name: str):
        model_provider_name = model_provider_name.lower()
        if model_provider_name == "gemini":
            return Gemini
        elif model_provider_name == "openai":
            return OpenAIV1Model
        elif model_provider_name == "openai-v2":
            return OpenAIV2Model
        elif model_provider_name == "groq":
            return GroqModel
        elif model_provider_name == "anthropic":
            return AnthropicModel
        elif model_provider_name == "huggingface" or model_provider_name == "hf":
            return HuggingFaceModel
        elif model_provider_name == "vllm":
            return VLLMModel
        elif model_provider_name == "openrouter":
            return OpenRouterModel
        else:
            raise Exception(f"Unknown model provider name {model_provider_name}")
