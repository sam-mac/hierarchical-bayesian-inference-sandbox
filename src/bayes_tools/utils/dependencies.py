"""Optional dependency helpers."""

import importlib.util


def module_available(module_name: str) -> bool:
    """Return ``True`` when ``module_name`` can be imported."""
    return importlib.util.find_spec(module_name) is not None
