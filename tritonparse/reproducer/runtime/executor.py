import subprocess
import sys
from typing import Dict, Optional
import os


def run_python(path: str, timeout: Optional[int] = None, env: Optional[Dict[str, str]] = None):
    # If custom environment variables are provided, merge them with the current environment
    process_env = os.environ.copy()
    if env:
        process_env.update(env)

    p = subprocess.Popen(
        [sys.executable, path],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=process_env,
    )
    try:
        out, err = p.communicate(timeout=timeout)
        return p.returncode, out, err
    except subprocess.TimeoutExpired:
        p.kill()
        # After killing, communicate again to get any remaining output
        out, err_after_kill = p.communicate()
        err = f"Process timed out after {timeout} seconds and was killed.\n{err_after_kill}"
        # A non-zero return code is appropriate for a timeout failure.
        return 1, out, err
