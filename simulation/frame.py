from dataclasses import dataclass

import numpy as np


@dataclass(slots=True, frozen=True)
class SimulationFrame:
    time: float

    positions: np.ndarray
    velocities: np.ndarray

    kinetic_energy: float
    potential_energy: float

    center_of_mass: np.ndarray

    @property
    def total_energy(self) -> float:
        return self.kinetic_energy + self.potential_energy
