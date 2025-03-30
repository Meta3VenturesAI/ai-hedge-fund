# --- START OF REVISED llm/models.py ---

import os
from langchain_anthropic import ChatAnthropic
from langchain_deepseek import ChatDeepSeek
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_community.chat_models import ChatOllama # For Ollama
#from langchain_mistralai import ChatMistralAI # <<<--- ADDED MISTRAL IMPORT
from langchain_core.language_models.chat_models import BaseChatModel # For type hinting

from enum import Enum
from pydantic import BaseModel
from typing import Tuple, List, Any

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

class ModelProvider(str, Enum):
    """Enum for supported LLM providers"""
    ANTHROPIC = "Anthropic"
    DEEPSEEK = "DeepSeek"
    GEMINI = "Gemini"
    GROQ = "Groq"
    OPENAI = "OpenAI"
    OLLAMA = "Ollama"
    MISTRAL = "MistralAI" # <<<--- ADDED MISTRAL PROVIDER

class LLMModel(BaseModel):
    """Represents an LLM model configuration"""
    display_name: str
    model_name: str # This should match the actual API model name (e.g., 'llama3', 'gpt-4o', 'mistral-large-latest')
    provider: ModelProvider

    def to_choice_tuple(self) -> Tuple[str, str, str]:
        """Convert to format needed for questionary choices"""
        return (self.display_name, self.model_name, self.provider.value)

    def has_json_mode(self) -> bool:
        """
        Check if the model provider/LangChain integration reliably supports
        JSON mode or structured output without needing manual extraction.
        Ollama, OpenAI, Groq, Anthropic, Mistral generally support structured output well via LangChain.
        DeepSeek and Gemini might require manual JSON extraction from markdown.
        """
        # Assume providers NOT in this list work well with structured output directly via LangChain
        return self.provider not in [ModelProvider.DEEPSEEK, ModelProvider.GEMINI]


# --- Define Available Models ---
# Add or remove models as needed. Ensure 'model_name' matches the API identifier.

AVAILABLE_MODELS: List[LLMModel] = [
    # --- Ollama Models ---
    LLMModel(
        display_name="[Ollama] Llama 3 Instruct",
        model_name="llama3",
        provider=ModelProvider.OLLAMA
    ),
    LLMModel(
        display_name="[Ollama] Mistral Instruct",
        model_name="mistral",
        provider=ModelProvider.OLLAMA
    ),
    LLMModel(
        display_name="[Ollama] Qwen 7B Chat",
        model_name="qwen", # e.g., 'qwen:7b-chat'
        provider=ModelProvider.OLLAMA
    ),
     LLMModel(
        display_name="[Ollama] Deepseek Coder V2 Lite",
        model_name="deepseek-coder-v2",
        provider=ModelProvider.OLLAMA
    ),
    LLMModel(
        display_name="[Ollama] Phi-3 Mini Instruct",
        model_name="phi3",
        provider=ModelProvider.OLLAMA
    ),
    LLMModel( # Example adding Code Llama
        display_name="[Ollama] Code Llama",
        model_name="codellama",
        provider=ModelProvider.OLLAMA
    ),
    # --- Add other Ollama models here ---

    # --- Anthropic Models ---
    LLMModel(
        display_name="[Anthropic] Claude 3.5 Haiku",
        model_name="claude-3-5-haiku-20240620",
        provider=ModelProvider.ANTHROPIC
    ),
    LLMModel(
        display_name="[Anthropic] Claude 3.5 Sonnet",
        model_name="claude-3-5-sonnet-20240620",
        provider=ModelProvider.ANTHROPIC
    ),

    # --- DeepSeek Models ---
    LLMModel(
        display_name="[DeepSeek] DeepSeek Coder V2",
        model_name="deepseek-coder",
        provider=ModelProvider.DEEPSEEK
    ),
    LLMModel(
        display_name="[DeepSeek] DeepSeek Chat V2",
        model_name="deepseek-chat",
        provider=ModelProvider.DEEPSEEK
    ),

    # --- Gemini Models ---
    LLMModel(
        display_name="[Gemini] Gemini 1.5 Flash",
        model_name="gemini-1.5-flash-latest",
        provider=ModelProvider.GEMINI
    ),
    LLMModel(
        display_name="[Gemini] Gemini 1.5 Pro",
        model_name="gemini-1.5-pro-latest",
        provider=ModelProvider.GEMINI
    ),

    # --- Groq Models ---
    LLMModel(
        display_name="[Groq] Llama 3 70B",
        model_name="llama3-70b-8192",
        provider=ModelProvider.GROQ
    ),
     LLMModel(
        display_name="[Groq] Llama 3 8B",
        model_name="llama3-8b-8192",
        provider=ModelProvider.GROQ
    ),
    LLMModel(
        display_name="[Groq] Mixtral 8x7B",
        model_name="mixtral-8x7b-32768",
        provider=ModelProvider.GROQ
    ),

    # --- Mistral AI Models --- # <<<--- ADDED MISTRAL SECTION
    LLMModel(
        display_name="[Mistral] Large",
        model_name="mistral-large-latest", # Check Mistral docs for exact identifier
        provider=ModelProvider.MISTRAL
    ),
    LLMModel(
        display_name="[Mistral] Small",
        model_name="mistral-small-latest", # Check Mistral docs for exact identifier
        provider=ModelProvider.MISTRAL
    ),
    # --- Add other Mistral models here if needed ---

    # --- OpenAI Models ---
    LLMModel(
        display_name="[OpenAI] GPT-4o",
        model_name="gpt-4o",
        provider=ModelProvider.OPENAI
    ),
    LLMModel(
        display_name="[OpenAI] GPT-4 Turbo",
        model_name="gpt-4-turbo",
        provider=ModelProvider.OPENAI
    ),
    LLMModel(
        display_name="[OpenAI] GPT-3.5 Turbo",
        model_name="gpt-3.5-turbo",
        provider=ModelProvider.OPENAI
    ),
]

# Create LLM_ORDER in the format expected by the UI
# Sort models alphabetically by display name for consistency
AVAILABLE_MODELS.sort(key=lambda x: x.display_name)
LLM_ORDER = [model.to_choice_tuple() for model in AVAILABLE_MODELS]

def get_model_info(model_name: str) -> LLMModel | None:
    """Get model information by model_name"""
    return next((model for model in AVAILABLE_MODELS if model.model_name == model_name), None)

def get_model(model_name: str, model_provider: ModelProvider) -> BaseChatModel | None:
    """
    Initializes and returns the LangChain chat model instance.

    Args:
        model_name: The name of the model (e.g., 'gpt-4o', 'llama3').
        model_provider: The ModelProvider enum value.

    Returns:
        An instance of a LangChain BaseChatModel or None if provider is unknown.

    Raises:
        ValueError: If the required API key for a provider is not found.
    """
    if model_provider == ModelProvider.GROQ:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            print(f"API Key Error: Please make sure GROQ_API_KEY is set in your .env file.")
            raise ValueError("Groq API key not found.")
        return ChatGroq(model=model_name, api_key=api_key, temperature=0)

    elif model_provider == ModelProvider.OPENAI:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            print(f"API Key Error: Please make sure OPENAI_API_KEY is set in your .env file.")
            raise ValueError("OpenAI API key not found.")
        return ChatOpenAI(model=model_name, api_key=api_key, temperature=0)

    elif model_provider == ModelProvider.ANTHROPIC:
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            print(f"API Key Error: Please make sure ANTHROPIC_API_KEY is set in your .env file.")
            raise ValueError("Anthropic API key not found.")
        return ChatAnthropic(model=model_name, api_key=api_key, temperature=0)

    elif model_provider == ModelProvider.DEEPSEEK:
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            print(f"API Key Error: Please make sure DEEPSEEK_API_KEY is set in your .env file.")
            raise ValueError("DeepSeek API key not found.")
        return ChatDeepSeek(model=model_name, api_key=api_key, temperature=0)

    elif model_provider == ModelProvider.GEMINI:
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            print(f"API Key Error: Please make sure GOOGLE_API_KEY is set in your .env file.")
            raise ValueError("Google API key not found.")
        return ChatGoogleGenerativeAI(model=model_name, google_api_key=api_key, temperature=0, convert_system_message_to_human=True)

    elif model_provider == ModelProvider.MISTRAL: # <<<--- ADDED MISTRAL BLOCK
        api_key = os.getenv("MISTRAL_API_KEY")
        if not api_key:
            print(f"API Key Error: Please make sure MISTRAL_API_KEY is set in your .env file.")
            raise ValueError("Mistral API key not found.")
        # Note: Import moved to top of file for convention
        return ChatMistralAI(model=model_name, api_key=api_key, temperature=0)

    elif model_provider == ModelProvider.OLLAMA:
        ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        print(f"Info: Using Ollama model '{model_name}' via {ollama_base_url}")
        try:
             return ChatOllama(model=model_name, base_url=ollama_base_url, temperature=0)
        except Exception as e:
            print(f"Error initializing Ollama: {e}. Is Ollama running and the model '{model_name}' pulled?")
            raise ValueError(f"Failed to initialize Ollama model '{model_name}'.")

    else:
        print(f"Error: Unknown model provider '{model_provider}'.")
        return None

# --- Optional: Add a function to check Ollama connectivity ---
def check_ollama_connection():
    """Checks if connection to Ollama server is possible and lists available models."""
    try:
        # Import locally to avoid hard dependency if Ollama isn't always used
        from langchain_community.chat_models import ChatOllama
        ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        # Use the underlying client to list models
        client = ChatOllama(base_url=ollama_base_url)._create_client()
        models = client.list()["models"] # Assuming the client has a 'list' method similar to ollama lib
        print(f"[Ollama Status] Connected to {ollama_base_url}. Available models:")
        for model in models:
            print(f"  - {model['name']}")
        return True
    except ImportError:
        print("[Ollama Status] Ollama library not found. Cannot check connection.")
        return False
    except Exception as e:
        print(f"[Ollama Status] Failed to connect to Ollama at {ollama_base_url}: {e}")
        print("  Ensure Ollama is running. Download from https://ollama.com/")
        return False

# Example: You could call check_ollama_connection() at the start of your main script.

# --- END OF REVISED llm/models.py ---
