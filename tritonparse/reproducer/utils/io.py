from pathlib import Path

def write_text(path: str, content: str, *, encoding="utf-8"):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(content, encoding=encoding)


