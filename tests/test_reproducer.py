import json
from pathlib import Path

from tritonparse.reproducer import (
    generate_allocation_snippet,
    generate_from_ndjson,
    generate_kwargs_dict,
)


class DummyProvider:
    def generate_code(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        temperature: float = 0.2,
        max_tokens: int = 8192,
        stop=None,
        extra=None,
    ) -> str:
        # Return a tiny script that always runs successfully
        return """
print("dummy ok")
"""


def _write_minimal_ndjson(path: Path) -> None:
    comp_event = {
        "event_type": "compilation",
        "payload": {
            "metadata": {
                "hash": "h1",
                "num_warps": 4,
                "num_stages": 2,
                "arch": "sm_90",
                "backend_name": "ptx",
                "triton_version": "3.0.0",
            },
            "python_source": {
                "code": """
import triton
import triton.language as tl

@triton.jit
def kernel(X_ptr):
    pass
""",
            },
        },
    }
    launch_event = {
        "event_type": "launch",
        "grid": [1],
        "compilation_metadata": {
            "hash": "h1",
            "num_warps": 4,
            "num_stages": 2,
        },
        "extracted_args": {
            "X": {
                "type": "tensor",
                "shape": [8, 8],
                "dtype": "float32",
                "device": "cuda:0",
                "stride": [8, 1],
                "is_contiguous": True,
                "numel": 64,
            },
            "BLOCK": {"type": "constexpr", "value": 128},
        },
    }
    with path.open("w", encoding="utf-8") as f:
        f.write(json.dumps(comp_event) + "\n")
        f.write(json.dumps(launch_event) + "\n")


def test_param_generator_snippet_basic(tmp_path):
    # Make a pseudo bundle and verify snippet contains expected constructs
    bundle = {
        "tensor_args": {
            "X": {
                "type": "tensor",
                "shape": [4, 4],
                "dtype": "float32",
                "device": "cuda:0",
                "stride": [4, 1],
                "is_contiguous": True,
            }
        },
        "args": {
            "X": {
                "type": "tensor",
                "shape": [4, 4],
                "dtype": "float32",
                "device": "cuda:0",
                "stride": [4, 1],
                "is_contiguous": True,
            },
            "N": {"type": "constexpr", "value": 4},
        },
    }
    snippet = generate_allocation_snippet(bundle)
    assert "torch.empty" in snippet
    assert "device = 'cuda:0'" in snippet
    kwargs = generate_kwargs_dict({"launch": {"kwargs": {"N": 4}}})
    assert kwargs == {"N": 4}


def test_orchestrator_with_dummy_provider(tmp_path):
    ndjson = tmp_path / "trace.ndjson"
    _write_minimal_ndjson(ndjson)

    out_py = tmp_path / "repro.py"
    res = generate_from_ndjson(
        str(ndjson),
        provider=DummyProvider(),
        launch_index=0,
        out_py=str(out_py),
        execute=True,
        retries=0,
        temperature=0.0,
        max_tokens=256,
    )
    assert out_py.exists()
    assert res.get("returncode", 0) == 0
    # stdout should contain our dummy output
    assert "dummy ok" in (res.get("stdout") or "")
