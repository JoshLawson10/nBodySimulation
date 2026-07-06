from __future__ import annotations

import csv
import math
import os
from dataclasses import dataclass

import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib import pyplot as plt

from data_types import Body, Vector3
from integration.rk45 import AdaptiveRK45Integrator
from physics.dynamics import NBodyDynamics
from physics.system import NBodySystem
from simulation.runner import SimulationRunner

G: float = 4 * math.pi**2
T_END: float = 5.0
SOFTENING: float = 1e-3
A_TOL: float = 1e-8
R_TOL: float = 1e-3
DT_MAX: float = 1e-3
DT_INITIAL: float = 1e-4
RESULTS_DIR: str = "results"

SOLAR_SYSTEM: dict[str, tuple[float, float, float]] = {
    # name, mass, x, vy
    "Sun": (1.0, 0.0, 0.0),
    "Mercury": (1.66012e-7, 0.387098, 10.084),
    "Venus": (2.44784e-6, 0.723332, 7.386),
    "Earth": (3.00349e-6, 1.000000, 2 * math.pi),
    "Mars": (3.22715e-7, 1.523679, 5.089),
    "Jupiter": (9.54792e-4, 5.2044, 2.755),
    "Saturn": (2.85886e-4, 9.5826, 2.029),
    "Uranus": (4.36624e-5, 19.2184, 1.433),
    "Neptune": (5.15139e-5, 30.1104, 1.144),
}

CONFIGURATIONS: list[tuple[str, list[str]]] = [
    ("01_reference", ["Sun", "Earth"]),
    ("02_+Mercury", ["Sun", "Earth", "Mercury"]),
    ("03_+Venus", ["Sun", "Earth", "Venus"]),
    ("04_+Mars", ["Sun", "Earth", "Mars"]),
    ("05_+Jupiter", ["Sun", "Earth", "Jupiter"]),
    ("06_+Saturn", ["Sun", "Earth", "Saturn"]),
    ("07_+Uranus", ["Sun", "Earth", "Uranus"]),
    ("08_+Neptune", ["Sun", "Earth", "Neptune"]),
    (
        "09_full_solar",
        [
            "Sun",
            "Earth",
            "Mercury",
            "Venus",
            "Mars",
            "Jupiter",
            "Saturn",
            "Uranus",
            "Neptune",
        ],
    ),
]

BODY_COLOURS: dict[str, str] = {
    "Sun": "#FDB813",
    "Mercury": "#b5b5b5",
    "Venus": "#e8cda0",
    "Earth": "#4fa3e0",
    "Mars": "#c1440e",
    "Jupiter": "#c88b3a",
    "Saturn": "#e4d191",
    "Uranus": "#7de8e8",
    "Neptune": "#4b70dd",
}

BODY_SIZES: dict[str, float] = {
    "Sun": 200.0,
    "Mercury": 30.0,
    "Venus": 50.0,
    "Earth": 50.0,
    "Mars": 40.0,
    "Jupiter": 120.0,
    "Saturn": 100.0,
    "Uranus": 70.0,
    "Neptune": 70.0,
}


@dataclass
class TimeseriesRow:
    t: float
    pos_dev: float
    vel_dev: float
    phase_dist: float
    eccentricity: float
    semi_major: float
    dE_rel: float


@dataclass
class SummaryRow:
    config: str
    n_bodies: int
    max_phase_dist: float
    mean_phase_dist: float
    max_dE_rel: float


def reference_state(t: float) -> tuple[np.ndarray, np.ndarray]:
    """Exact circular orbit of Earth at time t (years)."""
    a = 1.0
    omega = 2 * math.pi
    theta = omega * t
    r = np.array([a * math.cos(theta), a * math.sin(theta), 0.0])
    v = np.array([-a * omega * math.sin(theta), a * omega * math.cos(theta), 0.0])
    return r, v


def eccentricity_vector(r: np.ndarray, v: np.ndarray, gm: float) -> np.ndarray:
    """Laplace-Runge-Lenz eccentricity vector (Sun-relative)."""
    L = np.cross(r, v)
    return np.cross(v, L) / gm - r / np.linalg.norm(r)


def semi_major_axis(r: np.ndarray, v: np.ndarray, gm: float) -> float:
    """Vis-viva semi-major axis (AU). Returns nan if unbound."""
    r_mag = float(np.linalg.norm(r))
    v_sq = float(np.dot(v, v))
    denom = 2.0 / r_mag - v_sq / gm
    if abs(denom) < 1e-30:
        return float("nan")
    return 1.0 / denom


def make_bodies(names: list[str]) -> list[Body]:
    bodies = []
    for i, name in enumerate(names):
        mass, x, vy = SOLAR_SYSTEM[name]
        bodies.append(
            Body(
                id=i,
                mass=mass,
                position=Vector3(x, 0.0, 0.0),
                velocity=Vector3(0.0, vy, 0.0),
                name=name,
            )
        )
    return bodies


def make_runner(bodies: list[Body]) -> SimulationRunner:
    masses = np.array([b.mass for b in bodies], dtype=float)
    system = NBodySystem(masses=masses)
    dynamics = NBodyDynamics(system=system, softening=SOFTENING, G=G)
    integrator = AdaptiveRK45Integrator(a_tol=A_TOL, r_tol=R_TOL, dt_max=DT_MAX)
    initial_state = np.concatenate(
        [
            np.array(
                [[b.position.x, b.position.y, b.position.z] for b in bodies],
                dtype=float,
            ).ravel(),
            np.array(
                [[b.velocity.x, b.velocity.y, b.velocity.z] for b in bodies],
                dtype=float,
            ).ravel(),
        ]
    )
    return SimulationRunner(
        dynamics=dynamics,
        integrator=integrator,
        initial_state=initial_state,
        history_capacity=100_000,
    )


def save_trajectory_plot(
    config_name: str,
    body_names: list[str],
    all_positions: list[list[np.ndarray]],
) -> None:
    """Save a 2D xy-plane trajectory plot for this configuration."""
    sim_id = config_name.split("_")[0]
    sim_label = config_name.split("_", 1)[1].replace("_", " ").replace("+", "+ ")

    fig, ax = plt.subplots(figsize=(7, 7), facecolor="#0d0d0d")
    ax.set_facecolor("#0d0d0d")

    for i, name in enumerate(body_names):
        positions = np.array(all_positions[i])
        colour = BODY_COLOURS.get(name, "#ffffff")
        size = BODY_SIZES.get(name, 50.0)

        if name != "Sun":
            ax.plot(
                positions[:, 0],
                positions[:, 1],
                color=colour,
                linewidth=0.8,
                alpha=0.85,
            )

        ax.scatter(
            positions[-1, 0],
            positions[-1, 1],
            color=colour,
            s=size,
            zorder=5,
            label=name,
        )

        ax.scatter(
            positions[0, 0],
            positions[0, 1],
            color=colour,
            s=size * 0.4,
            marker="x",
            zorder=5,
        )

    ax.set_xlabel("$x$ (AU)", color="#cccccc", fontsize=10)
    ax.set_ylabel("$y$ (AU)", color="#cccccc", fontsize=10)
    ax.set_title(
        f"Simulation {sim_id}: {sim_label}  ($t = {T_END:.0f}$ yr)",
        color="#ffffff",
        fontsize=11,
    )
    ax.tick_params(colors="#cccccc")
    for spine in ax.spines.values():
        spine.set_color("#444444")
    ax.set_aspect("equal")
    ax.legend(
        loc="upper right",
        fontsize=7,
        facecolor="#1a1a1a",
        edgecolor="#444444",
        labelcolor="#cccccc",
    )
    ax.grid(True, alpha=0.15, color="#ffffff", linestyle="--")

    plot_path = os.path.join(RESULTS_DIR, f"trajectory_{config_name}.png")
    fig.savefig(plot_path, dpi=150, bbox_inches="tight", facecolor="#0d0d0d")
    plt.close(fig)


def run_configuration(
    config_name: str,
    body_names: list[str],
) -> SummaryRow:
    print(f"  [{config_name}]  {len(body_names)} bodies ... ", end="", flush=True)

    bodies = make_bodies(body_names)
    runner = make_runner(bodies)
    earth_idx = body_names.index("Earth")
    sun_idx = body_names.index("Sun")
    n = len(bodies)

    H0_k, H0_u = runner.dynamics.energies(runner.initial_state)
    H0 = H0_k + H0_u

    rows: list[TimeseriesRow] = []
    all_pos: list[list[np.ndarray]] = [[] for _ in body_names]

    for frame in runner.frames(dt=DT_INITIAL, t0=0.0):
        t = frame.time

        for i in range(n):
            all_pos[i].append(frame.positions[i].copy())

        r_rel = frame.positions[earth_idx] - frame.positions[sun_idx]
        v_rel = frame.velocities[earth_idx] - frame.velocities[sun_idx]

        r_ref, v_ref = reference_state(t)

        pos_dev = float(np.linalg.norm(r_rel - r_ref))
        vel_dev = float(np.linalg.norm(v_rel - v_ref))
        phase_dist = math.sqrt(pos_dev**2 + vel_dev**2)

        gm_sun = G * bodies[sun_idx].mass
        ecc = float(np.linalg.norm(eccentricity_vector(r_rel, v_rel, gm_sun)))
        sma = semi_major_axis(r_rel, v_rel, gm_sun)

        H_now = frame.kinetic_energy + frame.potential_energy
        dE_rel = abs(H_now - H0) / abs(H0) if abs(H0) > 1e-30 else 0.0

        rows.append(
            TimeseriesRow(
                t=t,
                pos_dev=pos_dev,
                vel_dev=vel_dev,
                phase_dist=phase_dist,
                eccentricity=ecc,
                semi_major=sma,
                dE_rel=dE_rel,
            )
        )

        if t >= T_END:
            break

    save_trajectory_plot(config_name, body_names, all_pos)

    ts_path = os.path.join(RESULTS_DIR, f"timeseries_{config_name}.csv")
    with open(ts_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "t_yr",
                "pos_dev_AU",
                "vel_dev_AU_yr",
                "phase_dist",
                "eccentricity",
                "semi_major_AU",
                "dE_rel",
            ]
        )
        for row in rows:
            writer.writerow(
                [
                    f"{row.t:.8f}",
                    f"{row.pos_dev:.6e}",
                    f"{row.vel_dev:.6e}",
                    f"{row.phase_dist:.6e}",
                    f"{row.eccentricity:.6e}",
                    f"{row.semi_major:.6f}",
                    f"{row.dE_rel:.6e}",
                ]
            )

    phase_dists = [r.phase_dist for r in rows]
    dE_rels = [r.dE_rel for r in rows]

    summary = SummaryRow(
        config=config_name,
        n_bodies=n,
        max_phase_dist=max(phase_dists),
        mean_phase_dist=sum(phase_dists) / len(phase_dists),
        max_dE_rel=max(dE_rels),
    )

    print(
        f"done  |  max d = {summary.max_phase_dist:.3e} AU  "
        f"max ΔE = {summary.max_dE_rel:.3e}"
    )

    return summary


def main() -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("\n" + "=" * 65)
    print("  SOLAR SYSTEM PERTURBATION INVESTIGATION")
    print(f"  T_END    = {T_END} yr  |  G = 4π²  (AU³ yr⁻² M☉⁻¹)")
    print(
        f"  a_tol    = {A_TOL:.0e}  |  r_tol = {R_TOL:.0e}  |  dt_max = {DT_MAX:.0e} yr"
    )
    print("=" * 65 + "\n")

    summaries: list[SummaryRow] = []

    for config_name, body_names in CONFIGURATIONS:
        summary = run_configuration(config_name, body_names)
        summaries.append(summary)

    summary_path = os.path.join(RESULTS_DIR, "summary.csv")
    with open(summary_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "config",
                "n_bodies",
                "max_phase_dist_AU",
                "mean_phase_dist_AU",
                "max_dE_rel",
            ]
        )
        for s in summaries:
            writer.writerow(
                [
                    s.config,
                    s.n_bodies,
                    f"{s.max_phase_dist:.6e}",
                    f"{s.mean_phase_dist:.6e}",
                    f"{s.max_dE_rel:.6e}",
                ]
            )

    print("\n" + "=" * 65)
    print(f"  {'Config':<24} {'n':>3}  {'max d (AU)':>12}  {'max ΔE_rel':>12}")
    print("  " + "-" * 57)
    for s in summaries:
        print(
            f"  {s.config:<24} {s.n_bodies:>3}  "
            f"{s.max_phase_dist:>12.3e}  "
            f"{s.max_dE_rel:>12.3e}"
        )
    print("=" * 65)
    print(f"\n  Results written to ./{RESULTS_DIR}/\n")


if __name__ == "__main__":
    main()
