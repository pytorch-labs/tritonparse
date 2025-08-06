# Copilot Instructions for TritonParse

## Project Overview
- **TritonParse** is a Python tool for visualizing and analyzing Triton kernel compilation and launch traces. It integrates tightly with Triton and PyTorch workflows, providing structured logging and IR comparison.
- The frontend is a React/Vite/TypeScript webapp (see `website/`), while the backend is Python (see `tritonparse/`).

## Key Components
- `tritonparse/structured_logging.py`: Core for logging kernel launches and compilation events, including stack traces and source mapping.
- `tritonparse/utils.py`: Main entry for parsing logs and generating output files (`.ndjson.gz`).
- `tritonparse/sourcemap_utils.py`: Utilities for stack trace analysis, session ID extraction, and source mapping.
- `tests/`: Contains both automated (unittest) and manual test scripts. See `tests/test_tritonparse.py` for main test suite.
- `website/`: Web UI for visualizing trace files. No backend server required; all processing is client-side.

## Developer Workflows
- **Build/Install**: `pip install -e .` (Python ≥3.10, Triton >3.3.1 required)
- **Generate Traces**: Use `tritonparse.structured_logging.init()` in your Triton/PyTorch code, then run `tritonparse.utils.unified_parse()` to produce trace files.
- **Run Tests**: `python -m unittest tests.test_tritonparse -v` (see `tests/README.md` for details)
- **Manual Test**: `python tests/test_add.py` (generates logs and parses them)
- **CI Scripts**: Use `.ci/*.sh` for environment setup, Triton install, and test runs. See `.ci/README.md` for workflow.

## Patterns & Conventions
- **Session ID Extraction**: Session IDs for autotune are derived from stack traces (see `get_autotune_session_id` in `sourcemap_utils.py`). The logic may need to distinguish user code from framework/runtime code.
- **Test Isolation**: Each test defines its own kernel to avoid cache interference. CUDA tests are skipped if no GPU is available.
- **Log Output**: All logs and parsed outputs are written to `tests/parsed_output/` or user-specified directories.
- **Environment Variables**: Commonly used for controlling debug, cache, and device selection (e.g., `TRITONPARSE_DEBUG=1`, `TORCHINDUCTOR_FX_GRAPH_CACHE=0`).

## Integration Points
- **Triton**: Must be installed from source for full feature support. See `pyproject.toml` and README for details.
- **PyTorch**: Used in tests and example kernels.
- **Web UI**: Consumes `.ndjson.gz` trace files for visualization. No server required.

## Troubleshooting
- If logs are missing, check Triton installation and ensure logging is enabled.
- For test failures, verify CUDA availability and correct environment setup.
- Use debug mode (`TRITONPARSE_DEBUG=1`) for verbose output.

## References
- See `README.md` (project root) for quick start, features, and documentation links.
- See `tests/README.md` for test structure and commands.
- See `.ci/README.md` for CI and environment setup scripts.

---

**For new agents:**
- Always check for the latest conventions in the above files before making changes.
- Prefer using provided utility functions and logging mechanisms over custom implementations.
- When analyzing stack traces or session IDs, refer to `sourcemap_utils.py` and related test cases for expected patterns.
