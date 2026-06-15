from dataclasses import dataclass

import numpy as np


@dataclass(slots=True, frozen=True)
class NBodySystem:
    masses: np.ndarray

    @property
    def body_count(self) -> int:
        return len(self.masses)

    @property
    def total_mass(self) -> float:
        return float(np.sum(self.masses))
