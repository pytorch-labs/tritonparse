import logging
from pathlib import Path
from typing import Any, Dict, Optional
import json
import subprocess
import sys
import textwrap
from datetime import datetime

from .ingestion.ndjson import build_context_bundle, find_launch_index_from_line
from .param_generator import generate_allocation_snippet, generate_kwargs_dict
from .prompts.loader import render_prompt
from .providers.base import LLMProvider
from .runtime.executor import run_python
from ..tools.prettify_ndjson import load_ndjson, save_prettified_json

logger = logging.getLogger(__name__)

TEMPLATE_PATH = Path(__file__).parent / "templates" / "reproducer_template.py"
# This should ideally be configurable, but for now, we mirror the template's value.
TRITON_KERNELS_PATH = (
    "/home/users/yhao24/.cache/huggingface/hub/models--kernels-community--"
    "triton_kernels/snapshots/1d2e9557ac0d4c651055a209055748d4db0fe65b/"
    "build/torch-universal/"
)


def _generate_import_statement(kernel_info: Dict[str, Any]) -> str:
    """Generates a Python import statement from kernel metadata."""
    file_path_str = kernel_info.get("file_path", "")
    function_name = kernel_info.get("function_name", "")

    if not file_path_str or not function_name:
        raise ValueError("Kernel file path or function name missing from context.")

    file_path = Path(file_path_str)
    
    # Ensure the path is within the known TRITON_KERNELS_PATH for relative import
    try:
        relative_path = file_path.relative_to(TRITON_KERNELS_PATH)
    except ValueError:
        logger.error(
            "Kernel path '%s' is not inside the expected TRITON_KERNELS_PATH '%s'.",
            file_path,
            TRITON_KERNELS_PATH,
        )
        # Fallback or error handling can be decided here. For now, raise.
        raise

    # Convert file path to module path
    # e.g., triton_kernels/topk.py -> triton_kernels.topk
    module_path = ".".join(relative_path.with_suffix("").parts)

    statement = (
        f"from {module_path} import {function_name} as imported_kernel_function"
    )
    logger.info("Generated import statement: %s", statement)
    return statement


def _parse_kernel_signature(kernel_source_code: str) -> tuple[list[str], list[str]]:
    """
    Parses a Triton kernel's source code to distinguish positional args
    from keyword args (those with default values).
    """
    signature_lines = []
    in_signature = False
    for line in kernel_source_code.splitlines():
        # Start capturing lines from 'def'
        if "def " in line:
            in_signature = True
        if in_signature:
            # Strip comments and leading/trailing whitespace
            clean_line = line.split("#")[0].strip()
            signature_lines.append(clean_line)
            # Stop capturing after the signature ends
            if "):" in line:
                break

    full_signature = "".join(signature_lines)
    # Extract content between the first '(' and the last '):'
    try:
        params_str = full_signature[
            full_signature.find("(") + 1 : full_signature.rfind("):")
        ]
    except IndexError:
        raise ValueError("Could not parse kernel signature.")

    # Clean up and split the parameters string
    params = [p.strip() for p in params_str.replace("\n", "").split(",") if p.strip()]

    positional_args = []
    keyword_args = []

    for param in params:
        if "=" in param:
            # Keyword arguments have a default value
            arg_name = param.split("=")[0].strip()
            keyword_args.append(arg_name)
        else:
            # Positional arguments do not have a default value
            arg_name = param.split(":")[0].strip()
            positional_args.append(arg_name)

    logger.debug("Parsed positional args: %s", positional_args)
    logger.debug("Parsed keyword args: %s", keyword_args)
    return positional_args, keyword_args


def _generate_invocation_snippet(
    positional_args: list[str], keyword_args: list[str]
) -> str:
    """Generates a single-line Python code snippet for kernel invocation."""
    # Prepare positional args for direct injection into the call
    pos_args_str = ", ".join([f'args_dict["{arg}"]' for arg in positional_args])

    # Prepare keyword args for direct injection
    kw_args_str = ", ".join([f'{arg}=args_dict["{arg}"]' for arg in keyword_args])

    # Combine them, ensuring proper comma separation
    all_args = []
    if pos_args_str:
        all_args.append(pos_args_str)
    if kw_args_str:
        all_args.append(kw_args_str)
    
    # Create the single-line call
    return f"imported_kernel_function[tuple(grid)]({', '.join(all_args)})"


def _find_line_for_launch_index(ndjson_path: str, launch_index: int) -> int:
    """Finds the 1-based line number for a given launch_index in an NDJSON file."""
    launch_count = 0
    with open(ndjson_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
                if event.get("event_type") == "launch":
                    if launch_count == launch_index:
                        logger.debug(
                            "Found launch_index %d on line %d.", launch_index, i
                        )
                        return i
                    launch_count += 1
            except json.JSONDecodeError:
                continue
    raise ValueError(f"Could not find launch_index {launch_index} in {ndjson_path}")


def _excerpt(s: str, n: int = 160):
    lines = s.splitlines()
    return "\n".join(lines[:n])


def generate_from_ndjson(
    ndjson_path: str,
    provider: Optional[LLMProvider],
    *,
    # Mode-specific params
    launch_index: int = 0,
    reproduce_error_from: Optional[str] = None,
    on_line: Optional[int] = None,
    attempts: int = 10,
    ai_analysis: bool = False,
    # Common params
    out_dir: Optional[str] = None,
    execute: bool = False,
    **gen_kwargs,
) -> Dict[str, Any]:
    is_error_repro_mode = reproduce_error_from is not None
    logger.info(
        "Starting reproducer generation in %s mode.",
        "Error Reproduction" if is_error_repro_mode else "Success",
    )

    # --- Build the base context bundle ---
    logger.info("Building context bundle from NDJSON...")
    # We need the kernel name early to determine the output path.
    # We determine the target launch index first.
    try:
        if is_error_repro_mode:
            if on_line is None:
                raise ValueError("on_line must be provided for error reproduction mode.")
            actual_launch_index = find_launch_index_from_line(ndjson_path, on_line)
        else:
            actual_launch_index = launch_index
    except ValueError as e:
        logger.error("Failed to determine target launch index: %s", e)
        raise

    bundle = build_context_bundle(ndjson_path, launch_index=actual_launch_index)
    context = bundle
    kernel_name = context.get("kernel_info", {}).get("function_name", "unknown_kernel")
    logger.debug("Context bundle created with keys: %s", list(context.keys()))

    # --- Step 1: Determine output paths ---
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    if out_dir:
        output_directory = Path(out_dir)
    else:
        # Default path: repro_output/<kernel_name>/
        output_directory = Path("repro_output") / kernel_name
    
    output_directory.mkdir(parents=True, exist_ok=True)
    
    out_py = output_directory / f"repro_{timestamp}.py"
    temp_json_path = output_directory / f"repro_context_{timestamp}.json"


    # --- Step 2: Determine target line and create single-event JSON ---
    try:
        if is_error_repro_mode:
            # on_line is already validated above
            target_line_num = on_line
        else:
            target_line_num = _find_line_for_launch_index(ndjson_path, launch_index)

        logger.info(
            "Targeting launch event at index %d on line %d.",
            actual_launch_index,
            target_line_num,
        )

        # Directly call the functions to extract the single launch event
        line_filter = {target_line_num}
        json_objects = load_ndjson(
            Path(ndjson_path), save_irs=True, line_filter=line_filter
        )
        if not json_objects:
            raise ValueError(f"No JSON object found on line {target_line_num}")

        save_prettified_json(json_objects, temp_json_path)
        logger.info("Successfully created single-event JSON at %s", temp_json_path)

    except (ValueError, FileNotFoundError) as e:
        logger.error("Failed to prepare single-event JSON context: %s", e)
        raise RuntimeError("Failed to prepare single-event JSON context") from e

    # --- Mode-specific logic ---
    exec_env = None
    exec_timeout = None  # Default to no timeout
    if is_error_repro_mode:
        full_error_text = Path(reproduce_error_from).read_text(encoding="utf-8")
        
        # Check for special conditions in the error log to adjust execution parameters.
        if "Assertion" in full_error_text:
            logger.info("Assertion error detected in original log. Setting TRITON_DEBUG=1.")
            exec_env = {"TRITON_DEBUG": "1"}
        
        if "hangs" in full_error_text.lower():
            logger.info("Hang detected in original log. Setting execution timeout to 30s.")
            exec_timeout = 30

        if ai_analysis:
            # Step 3: Get a structured error analysis from the LLM
            logger.info("Performing structured error analysis via LLM...")
            assert provider is not None, "Provider must be initialized for AI analysis."
            logger.debug("Full error text to be summarized:\n%s", full_error_text)
            summary_context = {"full_error_text": full_error_text}
            summary_prompt = render_prompt("summarize_error.txt", summary_context)
            # Use a simpler system prompt for the summary task
            summary_system_prompt = "You are an expert Triton debugging engineer."
            logger.debug(
                "Sending summarization prompt to LLM:\n---\n%s\n---\n%s\n---",
                summary_system_prompt,
                summary_prompt,
            )
            error_analysis_report = provider.generate_code(
                summary_system_prompt, summary_prompt, **gen_kwargs
            )
            context["error_analysis_report"] = error_analysis_report
            logger.debug("Received error analysis report:\n%s", error_analysis_report)

    # --- New Template-Based Generation ---
    logger.info("Loading reproducer template.")
    template_code = TEMPLATE_PATH.read_text(encoding="utf-8")

    # 1. Fill JSON Path
    # Use repr() to get a string literal with quotes, which is safe for code generation.
    final_code = template_code.replace("{{JSON_PATH_PLACEHOLDER}}", str(temp_json_path))

    # 2. Fill Kernel Import
    kernel_info = context.get("kernel_info", {})
    try:
        import_statement = _generate_import_statement(kernel_info)
    except (ValueError, FileNotFoundError) as e:
        logger.error("Failed to generate import statement: %s", e)
        # Handle error, maybe by returning a failed result
        raise RuntimeError(f"Could not generate import statement: {e}") from e

    final_code = final_code.replace("# {{KERNEL_IMPORT_PLACEHOLDER}}", import_statement)

    # 3. Fill Kernel Invocation using deterministic parsing
    logger.info("Generating kernel invocation snippet via signature parsing...")
    try:
        source_code = kernel_info.get("source_code", "")
        pos_args, kw_args = _parse_kernel_signature(source_code)
        invocation_snippet = _generate_invocation_snippet(pos_args, kw_args)
    except ValueError as e:
        logger.error("Failed to parse kernel signature or generate snippet: %s", e)
        raise RuntimeError("Failed to generate invocation snippet.") from e

    logger.debug("Generated invocation snippet:\n%s", invocation_snippet)

    final_code = final_code.replace(
        "# {{KERNEL_INVOCATION_PLACEHOLDER}}", invocation_snippet
    )

    out_py.write_text(final_code, encoding="utf-8")
    logger.info("Reproducer script saved to: %s", out_py)
    logger.debug("Generated code:\n%s", final_code)

    if not execute:
        return {"path": str(out_py)}

    # The execution, repair, and validation logic below remains largely the same,
    # but it will operate on the `final_code` generated from the template.
    # We will need to adapt the repair loop prompts later.

    # For now, let's comment out the old generation logic to avoid confusion.
    # The new logic is already in place above.

    # Old logic commented out:
    # system_prompt = render_prompt("system_import.txt", context)
    # ... rest of the old logic ...

    # Execute and optionally repair (SUCCESS-ONLY MODE)
    if not is_error_repro_mode:
        logger.info("Executing script in success mode (up to %d attempts)...", attempts)
        rc, out, err = run_python(str(out_py))
        # The first attempt is run once, subsequent attempts are retries
        retries_used = 0
        while rc != 0 and retries_used < attempts - 1:
            retries_used += 1
            logger.warning(
                "Execution failed. Attempting repair %d/%d...",
                retries_used,
                attempts - 1,
            )
            logger.debug("Stderr from failed execution:\n%s", err)
            # Build repair prompt
            repair_ctx = {
                "prev_code_excerpt": _excerpt(final_code, 200),
                "error_text": err[-4000:] if err else "(no stderr)",
            }
            # The repair prompt and logic will need to be updated for the template model
            repair_prompt = render_prompt("repair_loop.txt", repair_ctx) # This prompt might need updating
            logger.debug("Sending repair prompt to LLM:\n%s", repair_prompt)
            # This part needs to be smarter: it should only replace the invocation part
            assert provider is not None, "Provider must be initialized for repair loop."
            code_update = provider.generate_code("system_prompt_for_repair", repair_prompt, **gen_kwargs)
            # For now, we just re-generate the whole file for simplicity
            out_py.write_text(code_update, encoding="utf-8")
            final_code = code_update
            logger.info("Repair attempt %d complete. Re-executing.", retries_used)
            rc, out, err = run_python(str(out_py))

        if rc == 0:
            logger.info("Script executed successfully.")
        else:
            logger.error("Script failed after all repair attempts.")

        return {
            "path": str(out_py),
            "returncode": rc,
            "stdout": out,
            "stderr": err,
            "retries_used": retries_used,
        }
    else:
        # --- Error Reproduction Mode ---
        logger.info("Executing script in error-repro mode...")

        rc, out, err = run_python(str(out_py), timeout=exec_timeout, env=exec_env)
        logger.debug(
            "Execution finished with rc=%d.\nStdout:\n%s\nStderr:\n%s",
            rc,
            out,
            err,
        )

        if rc == 0:
            logger.warning(
                "Failed to reproduce the error. The script ran successfully."
            )
            return {
                "path": str(out_py),
                "returncode": rc,
                "stdout": out,
                "stderr": err,
                "message": "Failed to reproduce the error. The script ran successfully.",
            }

        # Script failed as expected.
        if not ai_analysis:
            logger.info(
                "AI verification skipped. Script failed as expected, marking as a successful reproduction."
            )
            return {
                "path": str(out_py),
                "returncode": rc,
                "stdout": out,
                "stderr": err,
                "message": "Successfully reproduced the failure (AI verification skipped).",
            }

        # With AI analysis, proceed to verify the error type.
        logger.info("Script failed as expected. Verifying error type via LLM...")
        assert provider is not None, "Provider must be initialized for AI verification."
        verify_ctx = {
            "target_error": context.get("error_analysis_report", ""),
            "actual_error": err,
        }
        verify_prompt = render_prompt("verify_error.txt", verify_ctx)
        verify_system_prompt = "You are a debugging assistant."
        verification_result = provider.generate_code(
            verify_system_prompt, verify_prompt, temperature=0.0
        ).strip()
        logger.debug("Verification result from LLM: %s", verification_result)

        if "YES" in verification_result:
            logger.info("Success! Script failed with the correct error type.")
            return {
                "path": str(out_py),
                "returncode": rc,
                "stdout": out,
                "stderr": err,
                "message": "Successfully reproduced the error.",
            }
        else:
            logger.warning("Script failed with an INCORRECT error type.")
            return {
                "path": str(out_py),
                "returncode": rc,
                "stdout": out,
                "stderr": err,
                "message": "Script failed, but with an incorrect error type.",
            }
