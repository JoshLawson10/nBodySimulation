import random
import time
from dataclasses import dataclass, field

import numpy as np
from matplotlib import pyplot as plt

N_BODIES = 10
G = 6.674e-11


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
    colour: str = "red"

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


bodies: list[Body] = []

for i in range(N_BODIES):
    mass = random.uniform(1e20, 1e30)
    position = Vector3(
        random.uniform(-1e11, 1e11),
        random.uniform(-1e11, 1e11),
        random.uniform(-1e11, 1e11),
    )
    velocity = Vector3(
        random.uniform(-1e4, 1e4),
        random.uniform(-1e4, 1e4),
        random.uniform(-1e4, 1e4),
    )
    bodies.append(Body(id=i, mass=mass, position=position, velocity=velocity))

ax1 = plt.figure().add_subplot(projection="3d")
ax1.set_title("3D Trajectories")
ax1.set_xlabel("x (m)")
ax1.set_ylabel("y (m)")
ax1.set_zlabel("z (m)")
ax1.legend(fontsize=7)

for body in bodies:
    ax1.scatter(
        body.position.x,
        body.position.y,
        body.position.z,
        label=body.name,
        color=body.colour,
    )

    ax1.text(
        body.position.x,
        body.position.y,
        body.position.z,
        body.name,
        fontsize=8,
        color=body.colour,
    )
plt.show()
