# TritonParse

[![License: BSD-3](https://img.shields.io/badge/License-BSD--3-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![GitHub Pages](https://img.shields.io/badge/GitHub%20Pages-Deploy-brightgreen)](https://pytorch-labs.github.io/tritonparse/)

A comprehensive visualization and analysis tool for Triton IR files, designed to help developers analyze, debug, and understand Triton kernel compilation processes.

## 🚀 Features

### Visualization & Analysis

- **Interactive Kernel Explorer**: Display detailed kernel information and stack traces
- **Multi-format IR Support**: View and explore multiple Triton IR formats:
  - TTGIR (Triton GPU IR)
  - TTIR (Triton IR)
  - LLIR (LLVM IR)
  - PTX (NVIDIA)
  - AMDGCN (AMD)
- **Side-by-side Comparison**: Compare the above IR code with synchronized highlighting
- **Interactive Code Views**: Click-to-highlight corresponding lines across different formats

### Structured Logging

- **Compilation Tracing**: Capture detailed Triton compilation events
- **Stack Trace Integration**: Full Python stack traces for compilation events
- **Metadata Extraction**: Comprehensive kernel metadata and compilation statistics
- **NDJSON Output**: Structured logging format for easy processing

### Website Deployment Options

- **GitHub Pages**: Automatic deployment with GitHub Actions
- **Local Development**: Full development environment setup

## 🛠️ Tech Stack

**Frontend:**

- React 19 with TypeScript
- Vite for build tooling
- Tailwind CSS for styling
- Monaco Editor for code display
- React Syntax Highlighter for syntax highlighting
- React Resizable Panels for layout

**Backend/Processing:**

- Python with Triton integration
- Structured logging and event tracing
- Source mapping extraction utilities

## 📦 Installation

### For Users (Trace Generation Only)

**Prerequisites:**

- **Python** >= 3.9
- **Triton** > 3.3.1

For now, you need to [manually compile latest Triton from source](https://github.com/triton-lang/triton?tab=readme-ov-file#install-from-source).

**Quick Start:**

```bash
# Clone the repository
git clone https://github.com/pytorch-labs/tritonparse.git
cd tritonparse

# Install Python dependencies
pip install -e .
```

### For Website Developers (Optional)

**Additional Prerequisites:**

- **Node.js** >= 18.0.0
- **npm**

**Website Setup:**

```bash
# Install website dependencies
cd website
npm install
```

## 🎯 Usage

### 1. Generate Triton Trace Files

Please refer to [wiki usage](https://github.com/pytorch-labs/tritonparse/wiki/Usage) for more details.

First, integrate TritonParse with your Triton/PyTorch code to generate trace files:

```python
import torch
# === TritonParse init ===
import tritonparse.structured_logging
# Initialize structured logging to capture Triton compilation events
# This will generate NDJSON trace logs in ./logs/
log_path = "./logs/"
tritonparse.structured_logging.init(log_path)
# === TritonParse init end ===

# The below is your original Triton/PyTorch 2 code
...

# === TritonParse parse ===
import tritonparse.utils
tritonparse.utils.unified_parse(log_path)
# === TritonParse parse end ===
```
See a full example in [`tests/test_add.py`](https://github.com/pytorch-labs/tritonparse/blob/main/tests/test_add.py).

Exampled output:
```bash
% TORCHINDUCTOR_FX_GRAPH_CACHE=0 python test_add.py
Triton kernel executed successfully
Torch compiled function executed successfully
WARNING:SourceMapping:No frame_id or frame_compile_id found in the payload.
WARNING:SourceMapping:No frame_id or frame_compile_id found in the payload.
tritonparse log file list: /tmp/tmpl1tp9fto/log_file_list.json
```
In our test example, it has two triton kernels: one is a pure triton kernel and the other is a PT2 compiled triton kernel. `TORCHINDUCTOR_FX_GRAPH_CACHE=0 ` is used to disable FX graph cache to let PT2 compiler compile the kernel every time. Otherwise, the final parsed log files will only contain the first triton kernel.
The final parsed gz files are stored in the `/tmp/tmpl1tp9fto/` directory. The `./logs` directory contains the raw NDJSON logs without source code mapping.

### 2. Analyze with Web Interface

#### Option A: Online Interface (Recommended)

**Visit [https://pytorch-labs.github.io/tritonparse/](https://pytorch-labs.github.io/tritonparse/)** to use the tool directly in your browser:

1. **Open your local trace file** (NDJSON or .gz format) directly in the browser
2. **Explore the visualization** using the Overview and Code Comparison tabs

**Supported File Formats:**
- `.ndjson` - Newline Delimited JSON trace files
- `.gz` - Gzip compressed trace files

#### Interface Overview

Once you load a trace file, you'll see the main interface with several key components:

**Kernel Overview & Details:**

![Kernel Overview](docs/screenshots/kernel-overview.png)

*The main interface showing the kernel list, compilation metadata, call stack, and navigation links to different IR representations.*

**Code Comparison View:**

![Code Comparison](docs/screenshots/code-comparison.png)

*Side-by-side comparison of different IR stages (e.g., TTGIR and PTX) with synchronized line highlighting and interactive navigation.*

#### Option B: Local Development (For Contributors)

For contributors working on the website:

```bash
cd website
npm install
npm run dev
```

Access the application at `http://localhost:5173`

**Available Scripts:**

- `npm run build` - Standard build
- `npm run build:single` - Standalone HTML file
- `npm run preview` - Preview production build

## 📁 Project Structure

```
tritonparse/
├── tritonparse/              # Python package
│   ├── structured_logging.py # Main logging infrastructure
│   ├── extract_source_mappings.py # Source mapping utilities
│   ├── source_type.py        # Source type definitions
│   ├── utils.py              # Helper utilities
│   ├── common.py             # Common functions
│   └── tp_logger.py          # Logger configuration
├── website/                  # React web application
│   ├── src/                  # React source code
│   ├── public/               # Static assets and example files
│   ├── scripts/              # Build utilities (inline-html.js)
│   ├── node_modules/         # Dependencies
│   ├── package.json          # Node.js dependencies
│   ├── vite.config.ts        # Vite configuration
│   └── dist/                 # Built application (after build)
├── docs/                     # Documentation and assets
│   ├── README.md             # Documentation guidelines
│   └── screenshots/          # Screenshots for README
├── tests/                    # Test files and example traces
│   ├── test_add.py           # Example Triton kernel test
│   ├── unit_tests.py         # Unit tests
│   └── *.ndjson              # Example trace files
├── run.py                    # Main runner script
├── pyproject.toml            # Python package configuration
├── LICENSE                   # BSD-3 license
├── CONTRIBUTING.md           # Contribution guidelines
└── CODE_OF_CONDUCT.md        # Code of conduct
```

## 🔧 Development

### Python Development

**Install in development mode:**

```bash
pip install -e .
```

**Example test:**

```bash
cd tests
python test_add.py
```

### Environment Variables

- `TRITONPARSE_DEBUG=1` - Enable debug logging
- `TRITONPARSE_NDJSON=1` - Output in NDJSON format (default)

### Website Development (For Contributors)

**Start development server:**

```bash
cd website
npm run dev
```

**Available Scripts:**

- `npm run dev` - Start development server
- `npm run build` - Production build
- `npm run build:single` - Standalone HTML build
- `npm run lint` - Run ESLint
- `npm run preview` - Preview production build

## 🚀 Deployment

### Live Website

The TritonParse visualization tool is automatically deployed and available at:
**[https://pytorch-labs.github.io/tritonparse/](https://pytorch-labs.github.io/tritonparse/)**

### For Contributors: Local Deployment

**Build standalone version:**

```bash
cd website
npm run build:single
```

The `dist/standalone.html` file contains the entire application and can be deployed anywhere.

## 📊 Understanding Triton Compilation

TritonParse helps visualize the Triton compilation pipeline:

1. **Python Source** → Triton kernel functions
2. **TTIR** → Triton's high-level IR
3. **TTGIR** → GPU-specific Triton IR
4. **LLIR** → LLVM IR representation
5. **PTX** → NVIDIA PTX assembly
6. **AMDGCN** → AMD GPU IR

Each stage can be inspected and compared to understand optimization transformations.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes
4. Run tests: `npm test` (website) and `python -m pytest` (Python)
5. Submit a pull request

## 📝 License

This project is licensed under the BSD-3 License - see the [LICENSE](LICENSE) file for details.

## 🔗 Related Projects

- [OpenAI Triton](https://github.com/openai/triton) - The Triton compiler and language
- [PyTorch](https://pytorch.org/) - Deep learning framework with Triton integration

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/pytorch-labs/tritonparse/issues)
- **Discussions**: [GitHub Discussions](https://github.com/pytorch-labs/tritonparse/discussions)
- **Wiki**: [TritonParse Wiki](https://github.com/pytorch-labs/tritonparse/wiki)

---

**Note**: This tool is designed for developers working with Triton kernels and GPU computing. Basic familiarity with CUDA, GPU programming concepts, and the Triton language is recommended for effective use.
