"""Reproducer subpackage: generate runnable Triton repro scripts from traces.

Contains:
- ingestion.ndjson: parse NDJSON and build a context bundle
- orchestrator: LLM-based code generation with optional execute/repair
- providers: LLM provider protocol and Gemini provider
- prompts: simple prompt loader and templates
- runtime.executor: helper to run generated Python scripts
- param_generator: synthesize tensor/scalar allocations to reduce LLM burden
"""

from .ingestion.ndjson import build_context_bundle
from .orchestrator import generate_from_ndjson
from .param_generator import generate_allocation_snippet, generate_kwargs_dict

__all__ = [
    "build_context_bundle",
    "generate_from_ndjson",
    "generate_allocation_snippet",
    "generate_kwargs_dict",
]


