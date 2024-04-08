import functools
import importlib
import os
import types
from typing import Any, cast, Dict, List, Optional, Sequence, Tuple


_BACKENDS: Dict[str, Any] = dict()


def register_backend(
    name: Optional[str] = None, tags: Sequence[str] = (),
):
    def wrapper(compiler_fn: Optional[Any] = None):
        if compiler_fn is None:
            return functools.partial(register_backend, name=name, tags=tags)
        assert callable(compiler_fn)
        fname = name or compiler_fn.__name__
        assert fname not in _BACKENDS, f"duplicate name: {fname}"
        _BACKENDS[fname] = compiler_fn
        compiler_fn._tags = tuple(tags)
        return compiler_fn

    return wrapper


def lookup_backend(compiler_fn):
    """Expand backend strings to functions"""
    if isinstance(compiler_fn, str):
        if compiler_fn not in _BACKENDS:
            _lazy_import()
        if compiler_fn not in _BACKENDS:
            raise RuntimeError(f"invalid backend {compiler_fn}")
        compiler_fn = _BACKENDS[compiler_fn]
    return compiler_fn


@functools.lru_cache(None)
def _lazy_import():
    from .. import backends

    def import_submodule(mod: types.ModuleType):
        """
        Ensure all the files in a given submodule are imported
        """
        for filename in sorted(os.listdir(os.path.dirname(cast(str, mod.__file__)))):
            if filename.endswith(".py") and filename[0] != "_":
                importlib.import_module(f"{mod.__name__}.{filename[:-3]}")

    import_submodule(backends)
