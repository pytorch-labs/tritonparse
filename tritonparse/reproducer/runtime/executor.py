import subprocess
import sys
from typing import Dict, Optional
import os


def run_python(path: str, timeout: int = 60, env: Optional[Dict[str, str]] = None):
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
    out, err = p.communicate(timeout=timeout)
    return p.returncode, out, err
