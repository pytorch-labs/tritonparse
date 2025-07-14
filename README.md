# TritonParse

[![License: BSD-3](https://img.shields.io/badge/License-BSD--3-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![GitHub Pages](https://img.shields.io/badge/GitHub%20Pages-Deploy-brightgreen)](https://pytorch-labs.github.io/tritonparse/)

**A comprehensive visualization and analysis tool for Triton IR files** — helping developers analyze, debug, and understand Triton kernel compilation processes.

🌐 **[Try it online →](https://pytorch-labs.github.io/tritonparse/?json_url=https%3A%2F%2Fpytorch-labs.github.io%2Ftritonparse%2Ff0_fc0_a0_cai-.ndjson)**

## ✨ Key Features

- **🔍 Interactive Visualization** - Explore Triton kernels with detailed metadata and stack traces
- **📊 Multi-format IR Support** - View TTGIR, TTIR, LLIR, PTX, and AMDGCN in one place
- **🔄 Side-by-side Comparison** - Compare IR stages with synchronized highlighting
- **📝 Structured Logging** - Capture detailed compilation events with source mapping
- **🌐 Ready-to-use Interface** - No installation required, works in your browser
- **🔒 Privacy-first** - All processing happens locally in your browser, no data uploaded

## 🚀 Quick Start

### 1. Generate Traces

```python
import tritonparse.structured_logging

# Initialize logging
tritonparse.structured_logging.init("./logs/")

# Your Triton/PyTorch code here
# ... your kernels ...

# Parse and generate trace files
import tritonparse.utils
tritonparse.utils.unified_parse("./logs/")
```
The example terminal output is:
```bash                      
tritonparse log file list: /tmp/tmp1gan7zky/log_file_list.json
INFO:tritonparse:Copying parsed logs from /tmp/tmp1gan7zky to /scratch/findhao/tritonparse/tests/parsed_output

================================================================================
📁 TRITONPARSE PARSING RESULTS
================================================================================
📂 Parsed files directory: /scratch/findhao/tritonparse/tests/parsed_output
📊 Total files generated: 2

📄 Generated files:
--------------------------------------------------
   1. 📝 dedicated_log_triton_trace_findhao__mapped.ndjson.gz (7.2KB)
   2. 📝 log_file_list.json (181B)
================================================================================
✅ Parsing completed successfully!
================================================================================
```

### 2. Visualize Results

**Visit [https://pytorch-labs.github.io/tritonparse/](https://pytorch-labs.github.io/tritonparse/?json_url=https%3A%2F%2Fpytorch-labs.github.io%2Ftritonparse%2Ff0_fc0_a0_cai-.ndjson)** and open your local trace files (.ndjson.gz format).

> **🔒 Privacy Note**: Your trace files are processed entirely in your browser - nothing is uploaded to any server! 

## 🛠️ Installation

**For basic usage (trace generation):**
```bash
git clone https://github.com/pytorch-labs/tritonparse.git
cd tritonparse
pip install -e .
```

**Prerequisites:** Python ≥ 3.10, Triton > 3.3.1 ([install from source](https://github.com/triton-lang/triton)), GPU required (NVIDIA/AMD)

## 📚 Complete Documentation

| 📖 Guide | Description |
|----------|-------------|
| **[🏠 Wiki Home](https://github.com/pytorch-labs/tritonparse/wiki)** | Complete documentation and navigation |
| **[📦 Installation Guide](https://github.com/pytorch-labs/tritonparse/wiki/01.-Installation)** | Detailed setup for all scenarios |
| **[📋 Usage Guide](https://github.com/pytorch-labs/tritonparse/wiki/02.-Usage-Guide)** | Complete workflow and examples |
| **[🌐 Web Interface Guide](https://github.com/pytorch-labs/tritonparse/wiki/03.-Web-Interface-Guide)** | Master the visualization interface |
| **[🔧 Developer Guide](https://github.com/pytorch-labs/tritonparse/wiki/04.-Developer-Guide)** | Contributing and development setup |
| **[❓ FAQ](https://github.com/pytorch-labs/tritonparse/wiki/06.-FAQ)** | Frequently asked questions |

## 🛠️ Tech Stack

- **Frontend**: React 19, TypeScript, Vite, Tailwind CSS, Monaco Editor
- **Backend**: Python with Triton integration, structured logging
- **Deployment**: GitHub Pages, automatic deployment

## 📊 Understanding Triton Compilation

TritonParse visualizes the complete Triton compilation pipeline:

**Python Source** → **TTIR** → **TTGIR** → **LLIR** → **PTX/AMDGCN**

Each stage can be inspected and compared to understand optimization transformations.

## 🤝 Contributing

We welcome contributions! Please see our **[Developer Guide](https://github.com/pytorch-labs/tritonparse/wiki/04.-Developer-Guide)** for:
- Development setup
- Code formatting standards  
- Pull request process
- Architecture overview

## 📞 Support & Community

- **🐛 Report Issues**: [GitHub Issues](https://github.com/pytorch-labs/tritonparse/issues)
- **💬 Discussions**: [GitHub Discussions](https://github.com/pytorch-labs/tritonparse/discussions)
- **📚 Documentation**: [TritonParse Wiki](https://github.com/pytorch-labs/tritonparse/wiki)

## 📄 License

This project is licensed under the BSD-3 License - see the [LICENSE](LICENSE) file for details.

---

**✨ Ready to get started?** Visit our **[Installation Guide](https://github.com/pytorch-labs/tritonparse/wiki/01.-Installation)** or try the **[online tool](https://pytorch-labs.github.io/tritonparse/)** directly!
