import logging
from pathlib import Path
from typing import Any, Dict, Optional

from .ingestion.ndjson import build_context_bundle, find_launch_index_from_line
from .param_generator import generate_allocation_snippet, generate_kwargs_dict
from .prompts.loader import render_prompt
from .providers.base import LLMProvider
from .runtime.executor import run_python

logger = logging.getLogger(__name__)


def _excerpt(s: str, n: int = 160):
    lines = s.splitlines()
    return "\n".join(lines[:n])


def generate_from_ndjson(
    ndjson_path: str,
    provider: LLMProvider,
    *,
    # Mode-specific params
    launch_index: int = 0,
    reproduce_error_from: Optional[str] = None,
    on_line: Optional[int] = None,
    attempts: int = 10,
    # Common params
    out_py: str = "repro.py",
    execute: bool = False,
    **gen_kwargs,
) -> Dict[str, Any]:
    is_error_repro_mode = reproduce_error_from is not None
    logger.info(
        "Starting reproducer generation in %s mode.",
        "Error Reproduction" if is_error_repro_mode else "Success",
    )

    if is_error_repro_mode:
        if on_line is None:
            raise ValueError("on_line must be provided for error reproduction mode.")
        # Step 2: Locate the launch_index from the line number
        logger.debug("Locating launch index for line %d", on_line)
        actual_launch_index = find_launch_index_from_line(ndjson_path, on_line)
        logger.debug("Found launch event at index %d", actual_launch_index)
    else:
        actual_launch_index = launch_index

    # --- Build the base context bundle ---
    logger.info("Building context bundle from NDJSON...")
    bundle = build_context_bundle(ndjson_path, launch_index=actual_launch_index)
    allocation_snippet = generate_allocation_snippet(bundle)
    kwargs_dict = generate_kwargs_dict(bundle)
    context = {
        **bundle,
        "allocation_snippet": allocation_snippet,
        "kwargs_dict": kwargs_dict,
    }
    logger.debug("Context bundle created with keys: %s", list(context.keys()))

    # --- Mode-specific logic ---
    if is_error_repro_mode:
        # Step 3: Get a structured error analysis from the LLM
        logger.info("Performing structured error analysis via LLM...")
        full_error_text = Path(reproduce_error_from).read_text(encoding="utf-8")
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

    system_prompt = render_prompt("system_import.txt", context)

    if is_error_repro_mode:
        user_prompt = render_prompt("reproduce_error.txt", context)
    else:
        user_prompt = render_prompt("generate_one_shot.txt", context)

    logger.info("Generating reproducer script via LLM (import strategy)...")
    logger.debug(
        "System prompt for code generation:\n---\n%s\n---", system_prompt
    )
    logger.debug("User prompt for code generation:\n---\n%s\n---", user_prompt)

    # --- Code Generation with Validation Loop ---
    code = ""
    for gen_attempt in range(3):  # Allow up to 3 tries to get valid code
        code = provider.generate_code(system_prompt, user_prompt, **gen_kwargs)
        if "{{" not in code:
            logger.debug("Generated code passed template validation on attempt %d.", gen_attempt + 1)
            break
        logger.warning(
            "Generated code contains un-rendered Jinja2 templates on attempt %d. Retrying.",
            gen_attempt + 1
        )
        # On the last attempt, fail hard
        if gen_attempt == 2:
            logger.error("Failed to generate valid code after 3 attempts. Aborting.")
            raise RuntimeError("LLM failed to render templates in generated code.")
    
    Path(out_py).write_text(code, encoding="utf-8")
    logger.info("Reproducer script saved to: %s", out_py)
    logger.debug("Generated code:\n%s", code)

    if not execute:
        return {"path": out_py}

    # Execute and optionally repair (SUCCESS-ONLY MODE)
    if not is_error_repro_mode:
        logger.info("Executing script in success mode (up to %d attempts)...", attempts)
        rc, out, err = run_python(out_py)
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
                "prev_code_excerpt": _excerpt(code, 200),
                "error_text": err[-4000:] if err else "(no stderr)",
            }
            repair_prompt = render_prompt("repair_loop.txt", repair_ctx)
            logger.debug("Sending repair prompt to LLM:\n%s", repair_prompt)
            code = provider.generate_code(system_prompt, repair_prompt, **gen_kwargs)
            Path(out_py).write_text(code, encoding="utf-8")
            logger.info("Repair attempt %d complete. Re-executing.", retries_used)
            rc, out, err = run_python(out_py)

        if rc == 0:
            logger.info("Script executed successfully.")
        else:
            logger.error("Script failed after all repair attempts.")

        return {
            "path": out_py,
            "returncode": rc,
            "stdout": out,
            "stderr": err,
            "retries_used": retries_used,
        }
    else:
        # --- Error Reproduction Mode ---
        # Step 5: Implement the failure-retry loop
        logger.info(
            "Executing script in error-repro mode (up to %d attempts)...", attempts
        )
        for attempt in range(attempts):
            logger.info("Attempt %d/%d to reproduce error...", attempt + 1, attempts)
            # The first attempt's code is already generated outside the loop
            if attempt > 0:
                # On subsequent attempts, generate new code
                logger.info("Previous attempt did not fail correctly. Generating new version...")
                retry_ctx = {
                    "error_analysis_report": context.get("error_analysis_report", ""),
                    "prev_code_excerpt": _excerpt(code, 200),
                    "last_actual_error": context.get("last_actual_error"),
                }
                retry_prompt = render_prompt("retry_reproduce_error.txt", retry_ctx)
                logger.debug(
                    "Sending retry prompt to LLM:\n---\n%s\n---", retry_prompt
                )
                
                # --- Code Generation with Validation Loop (inside retry) ---
                for gen_attempt in range(3):
                    code = provider.generate_code(system_prompt, retry_prompt, **gen_kwargs)
                    if "{{" not in code:
                        logger.debug("Generated code passed template validation on attempt %d.", gen_attempt + 1)
                        break
                    logger.warning(
                        "Generated code contains un-rendered Jinja2 templates on attempt %d. Retrying.",
                        gen_attempt + 1
                    )
                    if gen_attempt == 2:
                        logger.error("Failed to generate valid code after 3 attempts. Aborting.")
                        raise RuntimeError("LLM failed to render templates in generated code.")

                Path(out_py).write_text(code, encoding="utf-8")
                logger.debug(
                    "Generated new code for attempt %d:\n%s", attempt + 1, code
                )

            rc, out, err = run_python(out_py)
            logger.debug(
                "Execution of attempt %d finished with rc=%d.\nStdout:\n%s\nStderr:\n%s",
                attempt + 1,
                rc,
                out,
                err,
            )

            if rc != 0:
                # Script failed. Now, verify if it's the CORRECT failure.
                logger.info("Script failed. Verifying error type via LLM...")
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
                    # Success! The script failed as intended.
                    logger.info(
                        "Success! Script failed with the correct error type on attempt %d.",
                        attempt + 1,
                    )
                    return {
                        "path": out_py,
                        "returncode": rc,
                        "stdout": out,
                        "stderr": err,
                        "message": f"Successfully reproduced the error in {attempt + 1} attempt(s).",
                        "repro_attempts_used": attempt + 1,
                    }
                else:
                    logger.warning(
                        "Script failed with an INCORRECT error type on attempt %d.",
                        attempt + 1,
                    )
                    # This attempt failed, continue to the next retry.
                    context["last_actual_error"] = err  # Store for the next prompt

        # If the loop finishes, we failed to reproduce the error
        logger.warning(
            "Failed to reproduce the error after %d attempts. The final script ran successfully.",
            attempts,
        )
        return {
            "path": out_py,
            "returncode": 0,  # Last attempt was successful, which is failure for us
            "stdout": out,
            "stderr": err,
            "message": f"Failed to reproduce the error after {attempts} attempts. The last generated script ran successfully.",
            "repro_attempts_used": attempts,
        }
