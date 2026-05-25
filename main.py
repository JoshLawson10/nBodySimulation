import time
from dataclasses import dataclass

import numpy as np
from matplotlib import animation, rcParams
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

N_BODIES = 2
T_MAX = 3e9
STEPS_PER_FRAME = 20
HISTORY_CAPACITY = 50000

rcParams["figure.facecolor"] = "#0d0d0d"
rcParams["axes.facecolor"] = "#1a1a1a"
rcParams["axes.edgecolor"] = "#444444"
rcParams["text.color"] = "#f0f0f0"
rcParams["axes.labelcolor"] = "#f0f0f0"
rcParams["xtick.color"] = "#cccccc"
rcParams["ytick.color"] = "#cccccc"
rcParams["grid.color"] = "#333333"
rcParams["grid.alpha"] = 0.2
rcParams["font.family"] = "monospace"
rcParams["font.size"] = 9


@dataclass
class SimulationConfig:
    t0: float = 0.0
    t_max: float = T_MAX
    tol: float = 1e-12
    h0: float = 1e3
    h_min: float = 1e-3
    h_max: float = 1e6
    steps_per_frame: int = STEPS_PER_FRAME
    interval_ms: int = 30


def make_random_bodies(n: int, seed: int | None = 31) -> list[Body]:
    rng = np.random.default_rng(seed)
    cmap = plt.colormaps["hsv"]
    bodies: list[Body] = []

    for i in range(n):
        m = float(rng.uniform(1e29, 1e30))
        r = rng.uniform(-1e11, 1e11, 3)
        v = rng.uniform(-1e4, 1e4, 3)
        hue = i / max(n - 1, 1)
        rgb = cmap(hue)
        color = (
            f"#{int(rgb[0] * 255):02x}{int(rgb[1] * 255):02x}{int(rgb[2] * 255):02x}"
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

fig = plt.figure(figsize=(16, 10))
fig.suptitle(
    "N-BODY GRAVITATIONAL SIMULATION", fontsize=16, fontweight="bold", color="#00ff99"
)

gs = fig.add_gridspec(
    2, 2, hspace=0.3, wspace=0.3, left=0.05, right=0.98, top=0.93, bottom=0.08
)
ax_3d = fig.add_subplot(gs[:, 0], projection="3d")
ax_3d.set_facecolor("#1a1a1a")

ax_energy_dh = fig.add_subplot(gs[0, 1])
ax_energy_combined = fig.add_subplot(gs[1, 1])

ax_3d.set_title(
    "3D TRAJECTORIES", fontsize=12, fontweight="bold", pad=10, color="#00ff99"
)
ax_3d.set_xlabel("X (m)", fontsize=10, color="#00ccff")
ax_3d.set_ylabel("Y (m)", fontsize=10, color="#00ccff")
ax_3d.set_zlabel("Z (m)", fontsize=10, color="#00ccff")
ax_3d.grid(True, alpha=0.15, linestyle="--")

traj_lines = [
    Line3D(
        [],
        [],
        [],
        color=bodies[i].color,
        linewidth=1.5,
        label=f"Body {i + 1}",
        alpha=0.9,
    )
    for i in range(n)
]
body_markers = [
    Line3D(
        [],
        [],
        [],
        color=bodies[i].color,
        marker="o",
        markersize=10,
        linestyle="",
        markeredgecolor="#ffffff",
        markeredgewidth=0.5,
    )
    for i in range(n)
]

for line in traj_lines + body_markers:
    ax_3d.add_line(line)

ax_3d.legend(loc="upper left", fontsize=9, framealpha=0.9, fancybox=True)

ax_energy_dh.set_title(
    "ENERGY DEVIATION", fontsize=11, fontweight="bold", color="#00ff99"
)
(energy_dh_line,) = ax_energy_dh.plot([], [], color="#ff00ff", linewidth=2, label="ΔH")
ax_energy_dh.set_xlabel("Time (s)", fontsize=9)
ax_energy_dh.set_ylabel("ΔH (J)", fontsize=9, color="#ff00ff")
ax_energy_dh.grid(True, alpha=0.15)
ax_energy_dh.tick_params(axis="y", labelcolor="#ff00ff")

ax_energy_combined.set_title(
    "KINETIC & POTENTIAL", fontsize=11, fontweight="bold", color="#00ff99"
)
(kinetic_line,) = ax_energy_combined.plot(
    [], [], color="#00ccff", linewidth=2, label="Kinetic (T)"
)
(potential_line,) = ax_energy_combined.plot(
    [], [], color="#ff6600", linewidth=2, label="Potential (U)"
)
(total_line,) = ax_energy_combined.plot(
    [], [], color="#ff00ff", linewidth=2, label="Total (H)"
)
ax_energy_combined.set_xlabel("Time (s)", fontsize=9)
ax_energy_combined.set_ylabel("Energy (J)", fontsize=9)
ax_energy_combined.grid(True, alpha=0.15)
ax_energy_combined.legend(loc="best", fontsize=8)

info_text = fig.text(
    0.5,
    0.01,
    "Initialising...",
    ha="center",
    fontsize=10,
    family="monospace",
    bbox=dict(
        boxstyle="round", facecolor="#1a1a1a", edgecolor="#00ff99", linewidth=2, pad=0.8
    ),
    color="#f0f0f0",
)

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
        return (
            traj_lines
            + body_markers
            + [energy_dh_line, kinetic_line, potential_line, total_line]
        )

    actual_frames = frame_count
    ts_data = ts[:actual_frames]
    qs_data = qs_history[:, :actual_frames, :]

    for i in range(n):
        traj_lines[i].set_data(qs_data[i, :, 0], qs_data[i, :, 1])
        traj_lines[i].set_3d_properties(qs_data[i, :, 2])  # type: ignore[arg-type]
        body_markers[i].set_data([qs_data[i, -1, 0]], [qs_data[i, -1, 1]])
        body_markers[i].set_3d_properties(qs_data[i, -1, 2])  # type: ignore[arg-type]

    all_coords = qs_data.reshape(-1, 3)
    margin = 0.15 * np.ptp(all_coords, axis=0).max()
    for setter, idx in zip([ax_3d.set_xlim, ax_3d.set_ylim, ax_3d.set_zlim], [0, 1, 2]):
        setter(all_coords[:, idx].min() - margin, all_coords[:, idx].max() + margin)

    energy_dh_line.set_data(ts_data, Hs[:actual_frames])
    kinetic_line.set_data(ts_data, Ts[:actual_frames])
    potential_line.set_data(ts_data, Us[:actual_frames])
    total_line.set_data(ts_data, Hs[:actual_frames])

    for ax in [ax_energy_dh, ax_energy_combined]:
        ax.relim()
        ax.autoscale_view()

    wall_elapsed = time.time() - wall_t0
    progress = (ts_data[-1] - cfg.t0) / (cfg.t_max - cfg.t0) * 100
    status = "● COMPLETE" if finished else "● RUNNING"
    speed_ratio = ts_data[-1] / wall_elapsed if wall_elapsed > 0 else 0
    energy_error_pct = abs(Hs[actual_frames - 1]) / (H0 if H0 != 0 else 1) * 100

    status_str = (
        f"{status}  │  "
        f"Sim: {ts_data[-1]:.2e}s  │  "
        f"Real: {wall_elapsed:.1f}s  │  "
        f"Speed: {speed_ratio:.1f}x  │  "
        f"Progress: {progress:.1f}%  │  "
        f"ΔE: {Hs[actual_frames - 1]:.2e}J ({energy_error_pct:.3f}%)"
    )
    info_text.set_text(status_str)

    return (
        traj_lines
        + body_markers
        + [energy_dh_line, kinetic_line, potential_line, total_line]
    )


ani = animation.FuncAnimation(
    fig,
    update,
    interval=cfg.interval_ms,
    blit=False,
    cache_frame_data=False,
    save_count=0,
)
plt.show()
