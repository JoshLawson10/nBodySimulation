import time
from dataclasses import dataclass, field

import numpy as np
from matplotlib import animation
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3D

from data_types import Body, Vector3
from rk45 import solve_ivp_rk45
from utils import pack, unpack

"""
Naming Convention:
    position -> q
    displacement -> r
    velocity -> v
    acceleration -> a
"""


N_BODIES = 3


@dataclass
class SimulationConfig:
    t0: float = 0.0
    t_max: float = 3e9
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


def make_random_bodies(n: int, seed: int | None = 31) -> list[Body]:
    rng = np.random.default_rng(seed)
    cmap = plt.colormaps["tab10"]
    bodies: list[Body] = []

    for i in range(n):
        m = float(rng.uniform(1e20, 1e30))
        r = rng.uniform(-1e11, 1e11, 3)
        q = rng.uniform(-1e4, 1e4, 3)
        color = (
            f"#{int(cmap(i / max(n - 1, 1))[0] * 255):02x}"
            f"{int(cmap(i / max(n - 1, 1))[1] * 255):02x}"
            f"{int(cmap(i / max(n - 1, 1))[2] * 255):02x}"
        )
        bodies.append(
            Body(id=i, mass=m, position=Vector3(*r), velocity=Vector3(*q), color=color)
        )

    return bodies


cfg = SimulationConfig()
bodies = make_random_bodies(N_BODIES)
masses = np.array([b.mass for b in bodies])
n = len(bodies)
y0 = pack(bodies)

fig = plt.figure(figsize=(14, 6))
ax1 = fig.add_subplot(121, projection="3d")
ax2 = fig.add_subplot(122)

traj_lines = [
    Line3D([], [], [], color=bodies[i].color, linewidth=0.8, label=bodies[i].name)
    for i in range(n)
]
body_markers = [
    Line3D([], [], [], color=bodies[i].color, marker="o", markersize=6, linestyle="")
    for i in range(n)
]

for line in traj_lines + body_markers:
    ax1.add_line(line)

ax1.set_title("3D Trajectories")
ax1.set_xlabel("x (m)")
ax1.set_ylabel("y (m)")
ax1.set_zlabel("z (m)")
ax1.legend(fontsize=7)

(energy_line,) = ax2.plot([], [], color="orchid", linewidth=0.8)
(T_line,) = ax2.plot([], [], color="blue", linewidth=0.8)
(U_line,) = ax2.plot([], [], color="red", linewidth=0.8)
ax2.set_xlabel("Time (s)")
ax2.set_ylabel("ΔH (J)")
ax2.set_title("Energy Conservation")
ax2.grid(True, alpha=0.3)

plt.tight_layout()

state = SimulationState(qs=[[] for _ in range(n)])
solver = solve_ivp_rk45(
    y0=y0,
    t0=cfg.t0,
    t_max=cfg.t_max,
    h0=cfg.h0,
    h_min=cfg.h_min,
    h_max=cfg.h_max,
    masses=masses,
)


def update(frame: int):
    for _ in range(cfg.steps_per_frame):
        try:
            t, y = next(solver)
        except StopIteration:
            break

        q, _ = unpack(y, n)

        origin = q[0].copy()
        q_rel = q - origin

        state.ts.append(t)  # type: ignore[arg-type]

        for i in range(n):
            state.qs[i].append(Vector3(*q_rel[i]))

    if not state.qs[0]:
        return traj_lines + body_markers

    all_coords = []
    for i in range(n):
        arr = np.array([[v.x, v.y, v.z] for v in state.qs[i]])
        traj_lines[i].set_data(arr[:, 0], arr[:, 1])
        traj_lines[i].set_3d_properties(arr[:, 2])  # type: ignore[arg-type]
        body_markers[i].set_data([arr[-1, 0]], [arr[-1, 1]])
        body_markers[i].set_3d_properties(arr[-1, 2])  # type: ignore[arg-type]
        all_coords.append(arr)

    all_coords = np.concatenate(all_coords)
    margin = 0.1 * np.ptp(all_coords, axis=0).max()
    for setter, idx in zip([ax1.set_xlim, ax1.set_ylim, ax1.set_zlim], [0, 1, 2]):
        setter(all_coords[:, idx].min() - margin, all_coords[:, idx].max() + margin)

    return traj_lines + body_markers


ani = animation.FuncAnimation(
    fig,
    update,
    interval=cfg.interval_ms,
    blit=False,
    cache_frame_data=False,
    save_count=0,
)
plt.show()
