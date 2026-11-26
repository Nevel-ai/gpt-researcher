"""Register the real, dependency-light `gpt_researcher.credentials` package for offline tests.

The heavy `gpt_researcher` package itself is not imported: `from gpt_researcher.credentials import ...` resolves
the already registered submodule directly, the same way the budget tests register `gpt_researcher.utils.budget`.
"""
import importlib.util
from pathlib import Path
import sys

PACKAGE = Path(__file__).parents[1] / "gpt_researcher/credentials"


def load():
    module = sys.modules.get("gpt_researcher.credentials")
    if module is not None and getattr(module, "__file__", None) == str(PACKAGE / "__init__.py"):
        return module
    spec = importlib.util.spec_from_file_location("gpt_researcher.credentials", PACKAGE / "__init__.py",
                                                  submodule_search_locations=[str(PACKAGE)])
    module = importlib.util.module_from_spec(spec)
    sys.modules["gpt_researcher.credentials"] = module
    spec.loader.exec_module(module)
    # Submodules are registered up front: lazy submodule imports would otherwise try the absent parent package.
    for name in ("errors", "vault"):
        qualified = f"gpt_researcher.credentials.{name}"
        if qualified not in sys.modules:
            sub_spec = importlib.util.spec_from_file_location(qualified, PACKAGE / f"{name}.py")
            submodule = importlib.util.module_from_spec(sub_spec)
            sys.modules[qualified] = submodule
            sub_spec.loader.exec_module(submodule)
        setattr(module, name, sys.modules[qualified])
    return module


credentials = load()
