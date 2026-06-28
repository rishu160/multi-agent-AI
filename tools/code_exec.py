import sys
import io
import traceback
from langchain_core.tools import tool


@tool
def python_repl(code: str) -> str:
    """Execute Python code and return stdout + any error.
    
    Use this for calculations, data analysis, and algorithmic tasks.
    Only standard-library modules are available (no network calls).
    Always print() your final result so it appears in the output.
    """
    stdout_capture = io.StringIO()
    stderr_capture = io.StringIO()

    # Restrict builtins to prevent obviously dangerous calls
    restricted_globals = {
        "__builtins__": {
            k: v for k, v in __builtins__.items()  # type: ignore[union-attr]
            if k not in {"open", "exec", "eval", "compile", "__import__"}
        }
    } if isinstance(__builtins__, dict) else {"__builtins__": __builtins__}

    try:
        old_stdout, old_stderr = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = stdout_capture, stderr_capture
        exec(code, restricted_globals)  # noqa: S102
    except Exception:
        stderr_capture.write(traceback.format_exc())
    finally:
        sys.stdout, sys.stderr = old_stdout, old_stderr

    output = stdout_capture.getvalue()
    error = stderr_capture.getvalue()
    return output if output else (error or "No output produced.")
