"""Strict numeric coercion shared by UI and workflow boundaries."""

from __future__ import annotations

from decimal import Decimal, InvalidOperation
from typing import Any


def require_whole_number(value: Any, label: str, *, minimum: int) -> int:
    """Return an integer without silently truncating fractional input."""
    if value is None or isinstance(value, bool):
        raise ValueError(f"{label} must be a whole number.")
    if isinstance(value, str) and not value.strip():
        raise ValueError(f"{label} is empty. Enter a whole number.")
    try:
        number = Decimal(str(value))
    except (InvalidOperation, ValueError):
        raise ValueError(f"{label} must be a whole number.") from None
    if not number.is_finite() or number != number.to_integral_value():
        raise ValueError(f"{label} must be a whole number.")
    resolved = int(number)
    if resolved < minimum:
        raise ValueError(f"{label} must be at least {minimum}.")
    return resolved
