"""Provider factory for reproducer.

Currently supports Gemini only.
"""

from .config import load_config
from .providers.gemini import GeminiProvider


def make_gemini_provider() -> GeminiProvider:
    cfg = load_config()
    return GeminiProvider(
        project=cfg.project,
        location=cfg.location,
        model=cfg.model,
    )


