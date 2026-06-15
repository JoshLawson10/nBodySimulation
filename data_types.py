from dataclasses import dataclass

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
    size: float = 100.0

    def __post_init__(self):
        if not self.name:
            self.name = f"Body-{self.id}"

    def __repr__(self) -> str:
        return (
            f"Body('{self.name}', m={self.mass:.3e} kg, "
            f"q={self.position}, v={self.velocity})"
        )
