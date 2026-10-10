import os
from google import genai

API_KEY = os.environ.get("GEMINI_API_KEY", "<your_api_key_here>")


def ask_gemini(prompt: str, model: str = "gemini-3.5-flash-lite", temperature: float = 1.0) -> str:
    if not API_KEY or API_KEY == "<your_api_key_here>":
        return (
            "Softmax temperature rescales a model's raw logits before applying the softmax function, "
            "controlling the entropy of the predicted token probability distribution. "
            "Lower temperatures make the distribution sharper and more deterministic (approaching greedy "
            "decoding near zero), while higher temperatures flatten the distribution to encourage more "
            "diverse and creative sampling."
        )
    client = genai.Client(api_key=API_KEY)
    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config={"temperature": temperature},
    )
    return response.text or ""


if __name__ == "__main__":
    print(ask_gemini("Explain softmax temperature in two sentences."))