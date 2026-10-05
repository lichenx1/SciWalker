from __future__ import annotations

import os
from pathlib import Path
import signal
import subprocess


RUNTIME_ENVIRONMENT_KEYS = {
    "PATH",
    "LANG",
    "LC_ALL",
    "LC_CTYPE",
    "TZ",
    "LD_LIBRARY_PATH",
    "SYSTEMROOT",
    "WINDIR",
    "CUDA_VISIBLE_DEVICES",
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "OMP_THREAD_LIMIT",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "PYTHONHASHSEED",
}


def runtime_environment(workdir: Path) -> dict[str, str]:
    """Pass runtime settings without inheriting model/service credentials."""
    env = {key: value for key, value in os.environ.items() if key in RUNTIME_ENVIRONMENT_KEYS}
    env.update(PYTHONNOUSERSITE="1", PYTHONDONTWRITEBYTECODE="1", MPLBACKEND="Agg")
    for key, name in (
        ("MPLCONFIGDIR", "matplotlib"),
        ("NUMBA_CACHE_DIR", "numba"),
        ("XDG_CACHE_HOME", "cache"),
    ):
        directory = workdir / name
        directory.mkdir()
        env[key] = str(directory)
    return env


def execute_python(script: Path, timeout: int, python_executable: str) -> str:
    """Keep the established return format and terminate the test process group."""
    process = subprocess.Popen(
        [python_executable, str(script)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=script.parent,
        env=runtime_environment(script.parent),
        start_new_session=os.name == "posix",
    )
    timed_out = False
    try:
        stdout, stderr = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
    finally:
        try:
            if os.name == "posix":
                os.killpg(process.pid, signal.SIGKILL)
            elif process.poll() is None:
                process.kill()
        except ProcessLookupError:
            pass
        process.wait()
        if timed_out:
            stdout, stderr = process.communicate()
    if timed_out:
        return "timeout"
    if process.returncode:
        stderr = stderr.strip().replace("\n", " ")[:500]
        stdout = stdout.strip().replace("\n", " ")[:200]
        return f"returncode={process.returncode}; stderr={stderr}; stdout={stdout}"
    return ""
