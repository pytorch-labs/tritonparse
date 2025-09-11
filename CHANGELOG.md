# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2025-09-11

### TritonParse Release Notes (last 27 commits)

- **Date range**: 2025-07-25 — 2025-09-11
- **Scope**: Core library, website UI/UX, performance & scalability, CI/CD & packaging, documentation & maintenance.

### Highlights
- **Website usability**: Drag-and-drop to open logs; one-click copy in code viewers; sticky, compact kernel selector; footer shows app version, localized build date, and Git short SHA; tensor arguments in Launch Analysis now display concise summaries with expandable details.
- **Large-file parsing**: Streaming NDJSON parsing and robust gzip handling significantly reduce memory usage and improve stability for files >100 MB.
- **Core & integrations**: Persist Inductor kernel config into `inductor_metadata` and pass to JIT hooks; ensure Inductor path invokes `jit_post_compile_hook`; new `init_with_env` for environment-based initialization; move compilation timing `times` into `metadata` for automatic frontend rendering.
- **Releases & versioning**: Adopt setuptools-scm dynamic versioning; add Nightly PyPI publishing; enable stable publishing on tag push; fix nightly version potentially being older than stable; correct packaging license metadata.
- **CI stability**: Ubuntu 24.04 compatibility; improved CUDA/cuDNN setup and detection; parallelize jobs; add parallel CI for pip-installed Triton; better error visibility in install scripts; upgrade libstdc++.

### Changes by area
- **Core library**
  - Save Inductor kernel params to `inductor_metadata` and forward to JIT hooks.
  - Manually invoke `jit_post_compile_hook` in the Inductor Triton compile path.
  - Add `init_with_env` that reads `TRITON_TRACE_FOLDER` and `TRITON_TRACE_LAUNCH`.
  - Move compilation `times` into `metadata` so the frontend auto-renders it.
  - Use cached source in compile listener for stability.
  - Refactor source-mapping pipeline into modular units for maintainability.

- **Website UI/UX**
  - Drag-and-drop to open supported log files.
  - Copy button in code viewer panels.
  - Sticky/collapsible/compact kernel selector in Kernel Overview; resizable compilation stack trace vertically.
  - Launch Analysis: tensor args show concise summaries with expandable details.
  - Footer displays version, localized build date, and Git short SHA.
  - Streaming NDJSON parsing and improved error handling for large logs.

- **Performance & scalability**
  - Use streaming path for files >100 MB to reduce memory peaks and improve robustness.

- **CI/CD & packaging**
  - Enable setuptools-scm and nightly PyPI publishing.
  - Publish stable releases on tag push; improve version computation and tag detection.
  - Fix nightly version possibly lagging behind stable; add clear error on missing tags.
  - Add parallel CI for pip-installed Triton; recommend pip installation in docs.
  - Improve Ubuntu 24.04 setup, CUDA/cuDNN handling, and job parallelism.
  - Increase error visibility in install scripts and upgrade libstdc++.
  - Define lower bounds for prerequisites in `pyproject.toml`.

- **Docs & maintenance**
  - Move repository to `meta-pytorch` org; update links and guidance; add AI assistant context.
  - Update/restore CONTRIBUTING docs to avoid breaking downstream consumers.

- **Testing**
  - Preserve test outputs when `TEST_KEEP_OUTPUT=1` to aid debugging.

### Compatibility notes
- Versioning & publishing: setuptools-scm with tag-based stable releases and nightly dev versions. Ensure `PYPI_API_TOKEN` is configured in CI if publishing is intended.
- Data format: compilation timing `times` moved under `metadata`; update any downstream scripts that referenced the old location.
- Build metadata: footer shows localized build date and Git short SHA; restart dev server to refresh these values.

### Upgrade guidance
- Prefer Triton from PyPI (≥ 3.4.0) and adhere to the lower bounds declared in `pyproject.toml`.
- For deterministic build metadata in the website, set `BUILD_DATE` and `GIT_COMMIT_SHA_SHORT` in the environment when running dev/build.


## [0.1.1] - 2025-07-25

### Added

- **Launch Difference Analysis**: A new `launch_diff` event is automatically generated for each kernel, providing a concise summary of how launch parameters vary across different calls. This helps to quickly identify changes in kernel arguments, grid dimensions, and other metadata.
- **Enhanced Web UI for Launch Analysis**: The web interface now visualizes the `launch_diff` data, offering an interactive way to explore how kernel launches differ. It includes a detailed breakdown of constant vs. varying parameters and their value distributions.
- **Kernel-Centric Event Grouping**: The parser now intelligently groups compilation and launch events by kernel, making it easier to analyze the entire lifecycle of a specific kernel.
- **Launch Event Tracing Control**: Added an `enable_trace_launch` parameter to `tritonparse.structured_logging.init` to give users explicit control over whether to trace kernel launch events.
- **Enhanced Logging and Testing**: Improved the structured logging initialization and expanded test coverage to verify the correctness of `launch` and `compilation` event counts.

## [0.1.0] - 2025-07-21

This is the initial public release of TritonParse.

### Added

- **Interactive Web Interface**: A rich, client-side web UI for exploring, comparing, and understanding Triton IRs. Features side-by-side code views, synchronized highlighting, and detailed metadata panels.
- **Structured Logging Backend**: A powerful Python backend to capture detailed information from the Triton compiler and runtime, including IRs (TTIR, TTGIR, PTX, AMDGCN), metadata, timings, and Python source code, and outputs it as structured NDJSON logs.
- **Source-to-Source Mapping**: Automatic generation of bidirectional mappings between Python code and all intermediate representations (IRs), allowing you to trace a line of Python code all the way down to the generated assembly and back.
- **Kernel Launch Tracing**: Capability to trace each kernel launch, capturing the grid dimensions, kernel arguments (with detailed tensor information), and other runtime metadata.
- **Flexible Log Parsing CLI**: A command-line interface (`run.py`) to parse logs from local files or directories, and from single or multiple ranks in a distributed training job.
- **Prerequisites Documentation**: Clear requirements for Python (>=3.10), PyTorch, and Triton (>3.3.1, compiled from source).
- **Getting Started Guide**: A step-by-step workflow for generating, parsing, and visualizing traces.
- **Configuration via Environment Variables**: Support for `TRITON_TRACE`, `TRITON_TRACE_LAUNCH`, `TRITONPARSE_KERNEL_ALLOWLIST`, and `TRITON_TRACE_GZIP`.
