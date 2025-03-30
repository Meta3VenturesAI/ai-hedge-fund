# simple_test.py
import os
from dotenv import load_dotenv
from langchain_community.chat_models import ChatOllama
from langchain_deepseek import ChatDeepSeek

load_dotenv()

try:
    print("Testing Ollama...")
    ollama = ChatOllama(model="llama3", base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"), temperature=0)
    response_ollama = ollama.invoke("What is 1+1?")
    print(f"Ollama Response: {response_ollama}")
except Exception as e:
    print(f"Ollama Test Failed: {e}")

try:
    print("\nTesting DeepSeek...")
    deepseek = ChatDeepSeek(model="deepseek-chat", api_key=os.getenv("DEEPSEEK_API_KEY"), temperature=0)
    response_deepseek = deepseek.invoke("What is 1+1?")
    print(f"DeepSeek Response: {response_deepseek}")
except Exception as e:
    print(f"DeepSeek Test Failed: {e}")
