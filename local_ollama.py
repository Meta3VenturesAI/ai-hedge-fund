import subprocess

class LocalOllamaLLM:
    def __init__(self, model_name: str):
        self.model_name = model_name

    def call_model(self, prompt: str) -> str:
        """
        Calls the local Ollama model via CLI and returns the generated output.
        """
        command = ["ollama", "run", self.model_name, "--prompt", prompt]
        try:
            result = subprocess.run(command, capture_output=True, text=True, check=True)
            output = result.stdout.strip()
            return output
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Ollama call failed: {e.stderr}")

# Example usage:
if __name__ == "__main__":
    prompt = "Provide an overview of current market trends."
    model = LocalOllamaLLM(model_name="my-local-model")  # Replace with your installed model
    response = model.call_model(prompt)
    print("Response from Ollama:", response)

