from collections.abc import Generator
from dataclasses import dataclass

from mpl_toolkits.mplot3d.art3d import Line3D

from simulation.runner import SimulationRunner


@dataclass(slots=True)
class SimulationTrack:
    name: str
    runner: SimulationRunner
    generator: Generator
    linestyle: str
    alpha: float

    trajectories: list[Line3D]
    markers: list[Line3D]
