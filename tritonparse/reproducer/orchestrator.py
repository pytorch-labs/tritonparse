from pathlib import Path
from typing import Any, Dict

from .ingestion.ndjson import build_context_bundle
from .param_generator import generate_allocation_snippet, generate_kwargs_dict
from .prompts.loader import render_prompt
from .providers.base import LLMProvider
from .runtime.executor import run_python


def _excerpt(s: str, n: int = 160):
    lines = s.splitlines()
    return "\n".join(lines[:n])


def generate_from_ndjson(
    ndjson_path: str,
    provider: LLMProvider,
    *,
    launch_index=0,
    out_py="repro.py",
    execute=False,
    retries: int = 0,
    **gen_kwargs,
) -> Dict[str, Any]:
    bundle = build_context_bundle(ndjson_path, launch_index=launch_index)
    # Augment bundle with pre-generated parameter allocation code to reduce LLM burden
    allocation_snippet = generate_allocation_snippet(bundle)
    kwargs_dict = generate_kwargs_dict(bundle)
    context = {
        **bundle,
        "allocation_snippet": allocation_snippet,
        "kwargs_dict": kwargs_dict,
    }
    system_prompt = render_prompt("system.txt", context)
    user_prompt = render_prompt("generate_one_shot.txt", context)

    code = provider.generate_code(system_prompt, user_prompt, **gen_kwargs)
    Path(out_py).write_text(code, encoding="utf-8")

    if not execute:
        return {"path": out_py}

    # Execute and optionally repair
    rc, out, err = run_python(out_py)
    attempt = 0
    while rc != 0 and attempt < retries:
        attempt += 1
        # Build repair prompt
        repair_ctx = {
            "prev_code_excerpt": _excerpt(code, 200),
            "error_text": err[-4000:] if err else "(no stderr)",
        }
        repair_prompt = render_prompt("repair_loop.txt", repair_ctx)
        code = provider.generate_code(system_prompt, repair_prompt, **gen_kwargs)
        Path(out_py).write_text(code, encoding="utf-8")
        rc, out, err = run_python(out_py)

    return {
        "path": out_py,
        "returncode": rc,
        "stdout": out,
        "stderr": err,
        "retries_used": attempt,
    }
