# Welcome to TritonParse Wiki 🚀

**TritonParse** is a comprehensive visualization and analysis tool for Triton IR files, designed to help developers analyze, debug, and understand Triton kernel compilation processes.

[![License: BSD-3](https://img.shields.io/badge/License-BSD--3-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![GitHub Pages](https://img.shields.io/badge/GitHub%20Pages-Deploy-brightgreen)](https://pytorch-labs.github.io/tritonparse/)

## 🎯 Quick Navigation

### 📚 Getting Started
- **[Installation](01.-Installation)** - Complete setup instructions
- **[Quick Start Tutorial](#-quick-start)** - Your first TritonParse experience
- **[System Requirements](01.-Installation#-prerequisites)** - Prerequisites and compatibility

### 📖 User Guide
- **[Usage Guide](02.-Usage-Guide)** - Generate traces and analyze kernels
- **[Web Interface Guide](03.-Web-Interface-Guide)** - Master the visualization interface
- **[File Formats](02.-Usage-Guide#supported-file-formats)** - Understanding input/output formats
- **[Troubleshooting](01.-Installation#-troubleshooting)** - Common issues and solutions

### 🔧 Developer Guide
- **[Architecture Overview](04.-Developer-Guide#-architecture-overview)** - System design and components
- **[API Reference](04.-Developer-Guide#-api-reference)** - Python API documentation
- **[Contributing](04.-Developer-Guide#-contributing-guidelines)** - Development setup and guidelines
- **[Code Formatting](05.-Code-Formatting)** - Formatting standards and tools

### 🎓 Advanced Topics
- **[Source Mapping](02.-Usage-Guide#-understanding-the-results)** - IR stage mapping explained
- **[Environment Variables](01.-Installation#environment-variables)** - Configuration options
- **[Performance Tips](03.-Web-Interface-Guide#performance-considerations)** - Tips for large traces
- **[Custom Deployments](04.-Developer-Guide#deployment-options)** - Self-hosting and customization

### 📝 Examples & Reference
- **[Basic Examples](02.-Usage-Guide#example-complete-triton-kernel)** - Simple usage scenarios
- **[Advanced Examples](02.-Usage-Guide#-advanced-features)** - Complex use cases
- **[FAQ](06.-FAQ)** - Frequently asked questions
- **[Tech Stack](#-tech-stack)** - Technical terms and definitions

## 🌟 Key Features

### 🔍 Visualization & Analysis
- **Interactive Kernel Explorer** - Browse kernel information and stack traces
- **Multi-format IR Support** - View TTGIR, TTIR, LLIR, PTX, and AMDGCN
- **Side-by-side Comparison** - Compare IR stages with synchronized highlighting
- **Interactive Code Views** - Click-to-highlight corresponding lines

### 📊 Structured Logging
- **Compilation Tracing** - Capture detailed Triton compilation events
- **Stack Trace Integration** - Full Python stack traces for debugging
- **Metadata Extraction** - Comprehensive kernel metadata and statistics
- **NDJSON Output** - Structured logging format for easy processing

### 🌐 Deployment Options
- **GitHub Pages** - Ready-to-use online interface
- **Local Development** - Full development environment
- **Standalone HTML** - Self-contained deployments

## ⚡ Quick Start

### 1. Installation
```bash
# Clone the repository
git clone https://github.com/pytorch-labs/tritonparse.git
cd tritonparse

# Install dependencies
pip install -e .
```

### 2. Generate Traces
```python
import tritonparse.structured_logging

# Initialize logging
tritonparse.structured_logging.init("./logs/")

# Your Triton/PyTorch code here
...

# Parse logs
import tritonparse.utils
tritonparse.utils.unified_parse(source="./logs/", out="./parsed_output")
```

### 3. Analyze Results
Visit **[https://pytorch-labs.github.io/tritonparse/](https://pytorch-labs.github.io/tritonparse/)** and load your trace files!

## 🛠️ Tech Stack

**Frontend:** React 19, TypeScript, Vite, Tailwind CSS, Monaco Editor
**Backend:** Python, Triton integration, structured logging
**Deployment:** GitHub Pages, local development server

## 🔗 Important Links

- **Live Tool**: [https://pytorch-labs.github.io/tritonparse/](https://pytorch-labs.github.io/tritonparse/)
- **GitHub Repository**: [https://github.com/pytorch-labs/tritonparse](https://github.com/pytorch-labs/tritonparse)
- **Issues**: [GitHub Issues](https://github.com/pytorch-labs/tritonparse/issues)
- **Discussions**: [GitHub Discussions](https://github.com/pytorch-labs/tritonparse/discussions)

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](04.-Developer-Guide#-contributing-guidelines) for details on:
- Development setup
- Code formatting standards
- Pull request process
- Issue reporting

## 📄 License

This project is licensed under the BSD-3 License. See the [LICENSE](https://github.com/pytorch-labs/tritonparse/blob/main/LICENSE) file for details.

---

**Note**: This tool is designed for developers working with Triton kernels and GPU computing. Basic familiarity with GPU programming concepts (CUDA for NVIDIA or ROCm/HIP for AMD), and the Triton language is recommended for effective use. 
