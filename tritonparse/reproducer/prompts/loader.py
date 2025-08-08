import json
from pathlib import Path
from typing import Any, Dict

PROMPTS_DIR = Path(__file__).parent


def render_prompt(name: str, context: Dict[str, Any]) -> str:
    text = (PROMPTS_DIR / name).read_text(encoding="utf-8")
    # very simple {{key}} replacement for top-level keys; JSON for dicts
    for k, v in context.items():
        token = "{{ " + k + " }}"
        if token in text:
            if isinstance(v, (dict, list)):
                text = text.replace(token, json.dumps(v, ensure_ascii=False, indent=2))
            else:
                text = text.replace(token, str(v))
    return text
