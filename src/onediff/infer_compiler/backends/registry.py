import functools
import importlib
import os
import types
from typing import Any, cast, Dict, List, Optional, Sequence, Tuple


_BACKENDS: Dict[str, Any] = dict()


def register_backend(
    name: Optional[str] = None,
    tags: Sequence[str] = (),
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
            _lazy_import(compiler_fn)
        if compiler_fn not in _BACKENDS:
            raise RuntimeError(f"invalid backend {compiler_fn}")
        compiler_fn = _BACKENDS[compiler_fn]
    return compiler_fn


def _lazy_import(backend_name):
    from .. import backends

    backend_path = f"{backends.__name__}.{backend_name}"
    importlib.import_module(backend_path)
