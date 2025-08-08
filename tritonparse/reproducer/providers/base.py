from typing import Any, Dict, List, Optional, Protocol


class LLMProvider(Protocol):
    def generate_code(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        temperature: float = 0.2,
        max_tokens: int = 8192,
        stop: Optional[List[str]] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> str: ...
