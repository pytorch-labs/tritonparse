"""Provider factory for reproducer.

Currently supports Gemini only.
"""
import logging

from .config import load_config
from .providers.gemini import GeminiProvider

logger = logging.getLogger(__name__)


def make_gemini_provider() -> GeminiProvider:
    cfg = load_config()
    logger.info(
        "Creating GeminiProvider with project=%s, location=%s, model=%s",
        cfg.project,
        cfg.location,
        cfg.model,
    )
    return GeminiProvider(
        project=cfg.project,
        location=cfg.location,
        model=cfg.model,
    )
