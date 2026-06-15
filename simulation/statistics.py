from simulation.history import SimulationHistory


class SimulationStatistics:
    def __init__(
        self,
        history: SimulationHistory,
    ) -> None:
        self.history = history

    @property
    def initial_energy(self) -> float:
        return self.history.first.total_energy

    @property
    def current_energy(self) -> float:
        return self.history.latest.total_energy

    @property
    def energy_error(self) -> float:
        return self.current_energy - self.initial_energy

    @property
    def energy_error_percent(
        self,
    ) -> float:
        denominator = abs(self.initial_energy)

        if denominator == 0:
            return 0.0

        return abs(self.energy_error) / denominator * 100.0
