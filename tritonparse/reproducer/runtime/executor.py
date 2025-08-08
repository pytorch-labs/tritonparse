import subprocess, sys

def run_python(path: str, timeout: int = 60):
    p = subprocess.Popen([sys.executable, path], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    out, err = p.communicate(timeout=timeout)
    return p.returncode, out, err


