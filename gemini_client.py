import os
import google.generativeai as genai
from typing import List, Dict

class GeminiClient:
    def __init__(self, api_key: str = None, model: str = None):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise RuntimeError("GEMINI_API_KEY not set.")

        genai.configure(api_key=self.api_key)

        self.model_name = model or os.getenv("GEMINI_MODEL")

    def chat(self, messages: List[Dict], temperature: float = 0.2, max_tokens: int = 500) -> str:
        """
        Convert OpenAI-style messages into a prompt and send to Gemini.
        """
        
        prompt = "\n\n".join(
            f"{m.get('role', 'user').capitalize()}: {m.get('content', '')}"
            for m in messages
        )

        try:
            model = genai.GenerativeModel(self.model_name)
            resp = model.generate_content(
                prompt,
                generation_config={
                    "temperature": float(temperature),
                    "max_output_tokens": int(max_tokens),
                }
            )

            if hasattr(resp, "text") and resp.text:
                return resp.text.strip()

            if hasattr(resp, "candidates"):
                texts = []
                for c in resp.candidates:
                    if getattr(c, "content", None) and getattr(c.content, "parts", None):
                        texts.extend(
                            p.text for p in c.content.parts if getattr(p, "text", None)
                        )
                if texts:
                    return "\n".join(texts).strip()

            raise RuntimeError("Empty Gemini response.")

        except Exception as e:
            raise RuntimeError(f"Gemini API error: {e}")