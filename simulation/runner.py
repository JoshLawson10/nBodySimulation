from __future__ import annotations

from collections.abc import Generator

import numpy as np
from numpy.typing import NDArray

from integration.base import Integrator
from physics.dynamics import NBodyDynamics
from simulation.frame import SimulationFrame
from simulation.history import SimulationHistory
from simulation.statistics import (
    SimulationStatistics,
)


class SimulationRunner:
    def __init__(
        self,
        dynamics: NBodyDynamics,
        integrator: Integrator,
        initial_state: NDArray[np.float64],
        history_capacity: int = 20_000,
    ) -> None:
        self.dynamics = dynamics
        self.integrator = integrator
        self.initial_state = initial_state.copy()
        self.history = SimulationHistory(history_capacity)
        self.statistics = SimulationStatistics(self.history)

    def frames(
        self,
        *,
        dt: float,
        t0: float = 0.0,
    ) -> Generator[
        SimulationFrame,
        None,
        None,
    ]:
        n = self.dynamics.body_count
        coord_count = n * 3

        generator = self.integrator.solve(
            func=self.dynamics.derivatives,
            y0=self.initial_state,
            dt=dt,
            t0=t0,
        )

        for t, state in generator:
            positions = state[:coord_count].reshape(n, 3)
            velocities = state[coord_count:].reshape(n, 3)
            kinetic, potential = self.dynamics.energies(state)

            frame = SimulationFrame(
                time=t,
                positions=positions,
                velocities=velocities,
                kinetic_energy=kinetic,
                potential_energy=potential,
                center_of_mass=(self.dynamics.center_of_mass(positions)),
            )

            self.history.append(frame)

            yield frame
