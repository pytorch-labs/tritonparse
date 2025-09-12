from tritonparse.reproducer.ingestion.ndjson import load_input_file
from tritonparse.tp_logger import logger


def reproducer(
    input_path: str,
    line_index: int,
    out_dir: str,
):
    logger.debug(f"Building bundle from {input_path} at line {line_index}")
    events = load_input_file(input_path)
    logger.debug(f"Loaded {len(events)} events")
