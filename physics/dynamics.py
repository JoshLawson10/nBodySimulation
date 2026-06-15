from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from physics.system import NBodySystem


class NBodyDynamics:
    def __init__(
        self,
        system: NBodySystem,
        G: float = 1.0,
        softening: float = 0.0,
    ) -> None:
        self.system = system
        self.G = G
        self.softening = softening

    @property
    def body_count(self) -> int:
        return self.system.body_count

    def derivatives(
        self,
        t: float,
        *state_flat: float,
    ) -> np.ndarray:
        del t
        state = np.array(state_flat, dtype=float)
        positions = state[: self.body_count * 3].reshape(self.body_count, 3)
        velocities = state[self.body_count * 3 :].reshape(self.body_count, 3)
        accelerations = np.zeros_like(positions)

        for i in range(self.body_count):
            for j in range(i + 1, self.body_count):
                displacement = positions[j] - positions[i]
                distance_squared = (
                    float(np.dot(displacement, displacement)) + self.softening**2
                )
                distance = np.sqrt(distance_squared)
                factor = self.G / (distance_squared * distance)
                acc_ij = factor * displacement * self.system.masses[j]
                accelerations[i] += acc_ij
                accelerations[j] -= factor * displacement * self.system.masses[i]

        return np.concatenate(
            [
                velocities.ravel(),
                accelerations.ravel(),
            ]
        )

    def energies(
        self,
        state: NDArray[np.float64],
    ) -> tuple[float, float]:
        positions = state[: self.body_count * 3].reshape(self.body_count, 3)
        velocities = state[self.body_count * 3 :].reshape(self.body_count, 3)

        kinetic = float(0.5 * np.sum(self.system.masses[:, None] * velocities**2))
        potential = 0.0

        for i in range(self.body_count):
            for j in range(i + 1, self.body_count):
                displacement = positions[j] - positions[i]

                distance = np.sqrt(
                    np.dot(displacement, displacement) + self.softening**2
                )

                potential -= (
                    self.G * self.system.masses[i] * self.system.masses[j] / distance
                )

        return kinetic, potential

    def center_of_mass(
        self,
        positions: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        masses = self.system.masses

        return (
            np.sum(
                positions * masses[:, None],
                axis=0,
            )
            / self.system.total_mass
        )
