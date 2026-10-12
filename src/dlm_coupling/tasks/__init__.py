"""Deterministic, executable task families for the coupling pilot."""

from .arithmetic import generate_families, render_prompt, validate_family

__all__ = ["generate_families", "render_prompt", "validate_family"]
