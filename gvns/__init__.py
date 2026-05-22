"""gvns package initializer.

This file makes the `gvns` directory a proper Python package so
relative imports like `import gvns.operators` work from notebooks
and scripts that add the project root to `sys.path`.
"""

# Keep this file minimal to avoid side effects on import.
__all__ = []
