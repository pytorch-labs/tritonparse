# We'd like to sperate structured logging module and tritonparse module as much as possible. So, put the shared variables here.
import os

# The compilation information will be stored to /logs/DEFAULT_TRACE_FILE_PREFIX by default
# unless other flags disable or set another store. Add USER to avoid permission issues in shared servers.
DEFAULT_TRACE_FILE_PREFIX = (
    f"dedicated_log_triton_trace_{os.getenv('USER', 'unknown')}_"
)
DEFAULT_TRACE_FILE_PREFIX_WITHOUT_USER = "dedicated_log_triton_trace_"
