"""Small, hardware-independent helpers shared by scale clients and tests."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class StabilitySnapshot:
    """Result of evaluating the most recent stability window."""

    stable: bool
    sample_count: int
    window_size: int
    stability_range_g: float | None


class StabilityWindow:
    """A rolling max-minus-min stability gate for gram readings."""

    def __init__(self, window_size: int = 10, threshold_g: float = 2.0) -> None:
        if isinstance(window_size, bool) or window_size < 1:
            raise ValueError("window_size must be a positive integer")
        if int(window_size) != window_size:
            raise ValueError("window_size must be a positive integer")
        if not math.isfinite(threshold_g) or threshold_g < 0:
            raise ValueError("threshold_g must be finite and non-negative")
        self.window_size = int(window_size)
        self.threshold_g = float(threshold_g)
        self._values: deque[float] = deque(maxlen=self.window_size)

    def reset(self) -> None:
        self._values.clear()

    def add(self, weight_g: float) -> StabilitySnapshot:
        value = float(weight_g)
        if not math.isfinite(value):
            raise ValueError("weight_g must be finite")
        self._values.append(value)
        stability_range = None
        if self._values:
            stability_range = max(self._values) - min(self._values)
        stable = (
            len(self._values) == self.window_size
            and stability_range is not None
            and stability_range <= self.threshold_g
        )
        return StabilitySnapshot(
            stable=stable,
            sample_count=len(self._values),
            window_size=self.window_size,
            stability_range_g=stability_range,
        )


def weight_delta_g(current_g: float, previous_g: float) -> float:
    """Return the signed change from the previous to current weight."""

    current = float(current_g)
    previous = float(previous_g)
    if not math.isfinite(current) or not math.isfinite(previous):
        raise ValueError("weights must be finite")
    return current - previous
