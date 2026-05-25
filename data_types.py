import time
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Vector3:
    x: float
    y: float
    z: float

    def to_array(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z])

    def __repr__(self) -> str:
        return f"Vector3({self.x:.3e}, {self.y:.3e}, {self.z:.3e})"


@dataclass
class Body:
    id: int
    mass: float
    position: Vector3
    velocity: Vector3
    name: str = ""
    color: str = "red"

    def __post_init__(self):
        if not self.name:
            self.name = f"Body-{self.id}"

    def __repr__(self) -> str:
        return (
            f"Body('{self.name}', m={self.mass:.3e} kg, "
            f"q={self.position}, v={self.velocity})"
        )


@dataclass
class SimulationConfig:
    t_start: float = 0.0
    t_end: float = 3e9
    tol: float = 1e-8
    h0: float = 1e3
    h_min: float = 1e-3
    h_max: float = 1e6
    steps_per_frame: int = 20
    interval_ms: int = 30


@dataclass
class SimulationState:
    ts: list[float] = field(default_factory=list)
    qs: list[list[Vector3]] = field(default_factory=list)
    wall_t0: float = field(default_factory=time.time)
