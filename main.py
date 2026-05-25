import time
from dataclasses import dataclass

import numpy as np
from matplotlib import animation
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3D

from data_types import Body, Vector3
from rk45 import solve_ivp_rk45
from utils import pack, unpack

"""
Naming Convention:
    position → r (array of shape (n, 3))
    displacement → Δr
    velocity → v (array of shape (n, 3))
    acceleration → a (array of shape (n, 3))
"""


N_BODIES = 3
T_MAX = 3e9
STEPS_PER_FRAME = 20
HISTORY_CAPACITY = 50000


@dataclass
class SimulationConfig:
    t0: float = 0.0
    t_max: float = T_MAX
    tol: float = 1e-8
    h0: float = 1e3
    h_min: float = 1e-3
    h_max: float = 1e6
    steps_per_frame: int = STEPS_PER_FRAME
    interval_ms: int = 30


def make_random_bodies(n: int, seed: int | None = 31) -> list[Body]:
    rng = np.random.default_rng(seed)
    cmap = plt.colormaps["tab10"]
    bodies: list[Body] = []

    for i in range(n):
        m = float(rng.uniform(1e20, 1e30))
        r = rng.uniform(-1e11, 1e11, 3)
        v = rng.uniform(-1e4, 1e4, 3)
        color = (
            f"#{int(cmap(i / max(n - 1, 1))[0] * 255):02x}"
            f"{int(cmap(i / max(n - 1, 1))[1] * 255):02x}"
            f"{int(cmap(i / max(n - 1, 1))[2] * 255):02x}"
        )
        bodies.append(
            Body(id=i, mass=m, position=Vector3(*r), velocity=Vector3(*v), color=color)
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

info_text = fig.text(
    0.5, 0.01, "Initialising...", ha="center", fontsize=9, family="monospace"
)

plt.tight_layout()

ts = np.zeros(HISTORY_CAPACITY)
Hs = np.zeros(HISTORY_CAPACITY)
Ts = np.zeros(HISTORY_CAPACITY)
Us = np.zeros(HISTORY_CAPACITY)
qs_history = np.zeros((n, HISTORY_CAPACITY, 3))
frame_count = 0
H0 = None
wall_t0 = time.time()

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
    global frame_count, H0

    finished = False

    for _ in range(cfg.steps_per_frame):
        try:
            t, y, H = next(solver)
        except StopIteration:
            finished = True
            break

        positions, _ = unpack(y, n)
        T = H[0]
        U = H[1]

        origin = positions[0].copy()
        positions_rel = positions - origin

        if H0 is None:
            H0 = T + U

        ts[frame_count] = t
        Hs[frame_count] = T + U - H0
        Ts[frame_count] = T
        Us[frame_count] = U
        qs_history[:, frame_count, :] = positions_rel

        frame_count += 1

    if frame_count == 0:
        return traj_lines + body_markers + [energy_line, T_line, U_line]

    actual_frames = frame_count
    ts_data = ts[:actual_frames]
    qs_data = qs_history[:, :actual_frames, :]

    for i in range(n):
        traj_lines[i].set_data(qs_data[i, :, 0], qs_data[i, :, 1])
        traj_lines[i].set_3d_properties(qs_data[i, :, 2])  # type: ignore[arg-type]
        body_markers[i].set_data([qs_data[i, -1, 0]], [qs_data[i, -1, 1]])
        body_markers[i].set_3d_properties(qs_data[i, -1, 2])  # type: ignore[arg-type]

    all_coords = qs_data.reshape(-1, 3)
    margin = 0.1 * np.ptp(all_coords, axis=0).max()
    for setter, idx in zip([ax1.set_xlim, ax1.set_ylim, ax1.set_zlim], [0, 1, 2]):
        setter(all_coords[:, idx].min() - margin, all_coords[:, idx].max() + margin)

    energy_line.set_data(ts_data, Hs[:actual_frames])
    T_line.set_data(ts_data, Ts[:actual_frames])
    U_line.set_data(ts_data, Us[:actual_frames])
    ax2.relim()
    ax2.autoscale_view()

    wall_elapsed = time.time() - wall_t0
    progress = (ts_data[-1] - cfg.t0) / (cfg.t_max - cfg.t0) * 100
    status = "Complete" if finished else "Running"
    info_text.set_text(
        f"{status}  |  "
        f"Sim time: {ts_data[-1]:.3e} s  |  "
        f"Progress: {progress:.1f}%  |  "
        f"Wall time: {wall_elapsed:.1f} s  |  "
        f"ΔH: {Hs[actual_frames - 1]:.3e} J"
    )

    return traj_lines + body_markers + [energy_line, T_line, U_line]


ani = animation.FuncAnimation(
    fig,
    update,
    interval=cfg.interval_ms,
    blit=False,
    cache_frame_data=False,
    save_count=0,
)
plt.show()
