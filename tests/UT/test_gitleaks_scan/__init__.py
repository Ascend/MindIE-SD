"""Test package for the pre-commit gitleaks wrapper regression suite."""

import os


def _repo_root():
    """Absolute path of the repository root (three levels above this file)."""
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
