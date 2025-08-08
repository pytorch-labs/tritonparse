import os
from dataclasses import dataclass


@dataclass
class ReproducerConfig:
    project: str = os.getenv("GOOGLE_CLOUD_PROJECT", "")
    location: str = os.getenv("GOOGLE_LOCATION", "us-central1")
    model: str = os.getenv("TP_REPRO_MODEL", "gemini-2.5-pro")
    temperature: float = float(os.getenv("TP_REPRO_TEMPERATURE", "0.1"))
    max_tokens: int = int(os.getenv("TP_REPRO_MAX_TOKENS", "10000"))


def load_config() -> ReproducerConfig:
    return ReproducerConfig()
