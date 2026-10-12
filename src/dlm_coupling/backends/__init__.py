"""Backend factory. Importing it does not require torch or transformers."""

from typing import Any

from .base import Backend, PreparedCanvas
from .mock import MockBackend


def make_backend(config: dict[str, Any]) -> Backend:
    name = config.get("backend")
    if name is None:
        raise ValueError("Set backend explicitly to 'mock' or 'hf'; no implicit model substitution is allowed.")
    if name == "mock":
        if config.get("family", "mock") != "mock":
            raise ValueError("A mock backend must not be labelled as a real model family/checkpoint.")
        return MockBackend(config)
    if name == "hf":
        from .hf import HFBackend
        return HFBackend(config)
    raise ValueError(f"Unknown backend: {name!r}")


__all__ = ["Backend", "PreparedCanvas", "MockBackend", "make_backend"]
