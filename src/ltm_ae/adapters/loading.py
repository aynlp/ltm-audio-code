"""Explicit dynamic loading of locally installed external-model adapters."""

from __future__ import annotations

import importlib
from typing import Any

from ltm_ae.adapters.base import AudioTokenAdapter


def load_adapter(factory_spec: str, **kwargs: Any) -> AudioTokenAdapter:
    """Load an adapter factory in ``module.submodule:callable`` form.

    Keeping external integration in a caller-owned module prevents private
    checkouts, model paths, and credentials from entering this repository.
    """

    if ":" not in factory_spec:
        raise ValueError("Adapter factory must use 'module.submodule:callable' syntax.")
    module_name, attribute = factory_spec.split(":", maxsplit=1)
    factory = getattr(importlib.import_module(module_name), attribute, None)
    if factory is None or not callable(factory):
        raise ValueError(f"Adapter factory {factory_spec!r} is not callable.")
    adapter = factory(**kwargs)
    if not isinstance(adapter, AudioTokenAdapter):
        raise TypeError(f"Adapter factory {factory_spec!r} did not return an AudioTokenAdapter.")
    return adapter
