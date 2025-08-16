import logging
import os
import re
from typing import Any, Dict, List, Optional

from google.genai import Client

logger = logging.getLogger(__name__)


def _extract_python_block(s: str) -> str:
    m = re.search(r"""```python\s+(.*?)```""", s, flags=re.S)
    if m:
        logger.debug("Extracted Python code block from markdown.")
        return m.group(1).strip()
    logger.debug("No markdown code block found; returning raw text.")
    return ""


class GeminiProvider:
    def __init__(
        self, project: str, location: str = "us-central1", model: str = "gemini-2.5-pro"
    ):
        # Expect GOOGLE_APPLICATIONS_CREDENTIALS to be set
        if not os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
            raise EnvironmentError("GOOGLE_APPLICATION_CREDENTIALS not set.")
        self.client = Client(vertexai=True, project=project, location=location)
        self.model = model
        logger.debug("GeminiProvider initialized for model: %s", model)

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
        config = {
            "temperature": temperature,
            "max_output_tokens": max_tokens,
        }
        logger.info("Sending request to Gemini model '%s'...", self.model)
        logger.debug(
            "Gemini request config: %s",
            config,
        )
        resp = self.client.models.generate_content(
            model=self.model,
            contents=full_prompt,
            config=config,
        )
        text = getattr(resp, "text", "") or ""
        logger.debug("Received raw response from Gemini:\n%s", _excerpt(text, 400))
        code = _extract_python_block(text) or text
        if not code.strip():
            logger.error("Empty response from Gemini. Raw text: %s", text[:2000])
            raise RuntimeError(f"Empty response from Gemini. Raw: {text[:2000]}")
        return code


def _excerpt(s: str, n: int) -> str:
    """Helper to truncate long strings for logging."""
    return s[:n] + "..." if len(s) > n else s
