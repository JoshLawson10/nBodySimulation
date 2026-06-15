from collections import deque

from simulation.frame import SimulationFrame


class SimulationHistory:
    def __init__(self, capacity: int) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")

        self._frames: deque[SimulationFrame] = deque(maxlen=capacity)

    def append(self, frame: SimulationFrame) -> None:
        self._frames.append(frame)

    @property
    def frames(self) -> tuple[SimulationFrame, ...]:
        return tuple(self._frames)

    @property
    def latest(self) -> SimulationFrame:
        if not self._frames:
            raise RuntimeError("history is empty")

        return self._frames[-1]

    @property
    def first(self) -> SimulationFrame:
        if not self._frames:
            raise RuntimeError("history is empty")

        return self._frames[0]

    def __len__(self) -> int:
        return len(self._frames)
