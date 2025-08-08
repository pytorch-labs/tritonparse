import os
import re
from typing import Any, Dict, List, Optional

from google.genai import Client


def _extract_python_block(s: str) -> str:
    m = re.search(r"""```python\s+(.*?)```""", s, flags=re.S)
    return m.group(1).strip() if m else ""


class GeminiProvider:
    def __init__(
        self, project: str, location: str = "us-central1", model: str = "gemini-2.5-pro"
    ):
        # Expect GOOGLE_APPLICATIONS_CREDENTIALS to be set
        if not os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
            raise EnvironmentError("GOOGLE_APPLICATION_CREDENTIALS not set.")
        self.client = Client(vertexai=True, project=project, location=location)
        self.model = model

    def generate_code(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        temperature: float = 0.2,
        max_tokens: int = 8192,
        stop: Optional[List[str]] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> str:
        # Gemini doesn't have a 'system' role in this SDK, prepend system to user
        full_prompt = f"{system_prompt.strip()}\n\n---\n\n{user_prompt.strip()}"
        resp = self.client.models.generate_content(
            model=self.model,
            contents=full_prompt,
            config={
                "temperature": temperature,
                "max_output_tokens": max_tokens,
            },
        )
        text = getattr(resp, "text", "") or ""
        code = _extract_python_block(text) or text
        if not code.strip():
            raise RuntimeError(f"Empty response from Gemini. Raw: {text[:2000]}")
        return code
