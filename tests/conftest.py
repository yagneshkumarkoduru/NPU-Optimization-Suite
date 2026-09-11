"""Shared test configuration: exposes repo modules to pytest via importlib.

The implementation engines live in directories without package __init__
files, so each module is loaded directly from its file path. Module names
are unique across the repo, which keeps sys.modules registration safe.
"""

import importlib.util
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def load_module(relative_path: str, module_name: str):
    """Load a repository python module from its path relative to the repo root."""
    module_path = REPO_ROOT / relative_path
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module
