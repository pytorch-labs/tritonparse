# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2024-07-31

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