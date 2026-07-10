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

G: float = 4 * math.pi**2  # AU³ yr⁻² M☉⁻¹
T_END: float = 50.0  # years
SOFTENING: float = 1e-5  # AU
A_TOL: float = 1e-8
R_TOL: float = 1e-3
DT_MAX: float = 1e-3  # yr
DT_INITIAL: float = 1e-4  # yr
RESULTS_DIR: str = "results"

SOLAR_SYSTEM: dict[str, tuple[float, float, float]] = {
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

BG = "#0d0d0d"
GRID_C = "#2a2a2a"
TEXT_C = "#cccccc"
TITLE_C = "#ffffff"

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

CONFIG_COLOURS: dict[str, str] = {
    "01_reference": "#aaaaaa",
    "02_+Mercury": "#b5b5b5",
    "03_+Venus": "#e8cda0",
    "04_+Mars": "#c1440e",
    "05_+Jupiter": "#c88b3a",
    "06_+Saturn": "#e4d191",
    "07_+Uranus": "#7de8e8",
    "08_+Neptune": "#4b70dd",
    "09_full_solar": "#ff6b9d",
}

CONFIG_LABELS: dict[str, str] = {
    "01_reference": "01 Reference",
    "02_+Mercury": "02 + Mercury",
    "03_+Venus": "03 + Venus",
    "04_+Mars": "04 + Mars",
    "05_+Jupiter": "05 + Jupiter",
    "06_+Saturn": "06 + Saturn",
    "07_+Uranus": "07 + Uranus",
    "08_+Neptune": "08 + Neptune",
    "09_full_solar": "09 Full Solar System",
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
    max_eccentricity: float
    mean_eccentricity: float
    final_eccentricity: float
    max_dE_rel: float


def dark_axes(ax) -> None:
    ax.set_facecolor(BG)
    ax.tick_params(colors=TEXT_C, labelsize=8)
    ax.xaxis.label.set_color(TEXT_C)
    ax.yaxis.label.set_color(TEXT_C)
    for spine in ax.spines.values():
        spine.set_color("#444444")
    ax.grid(True, color=GRID_C, linewidth=0.5, linestyle="--")


def dark_fig(figsize: tuple[float, float]) -> tuple:
    fig, ax = plt.subplots(figsize=figsize, facecolor=BG)
    dark_axes(ax)
    return fig, ax


def savefig(fig, filename: str) -> None:
    path = os.path.join(RESULTS_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)


def legend(ax, fontsize: int = 7, **kwargs) -> None:
    ax.legend(
        fontsize=fontsize,
        facecolor="#1a1a1a",
        edgecolor="#444444",
        labelcolor=TEXT_C,
        **kwargs,
    )


def reference_state(t: float) -> tuple[np.ndarray, np.ndarray]:
    """Exact circular orbit of Earth at time t (years)."""
    a = 1.0
    omega = 2 * math.pi
    theta = omega * t
    r = np.array([a * math.cos(theta), a * math.sin(theta), 0.0])
    v = np.array([-a * omega * math.sin(theta), a * omega * math.cos(theta), 0.0])
    return r, v


def eccentricity_vector(r: np.ndarray, v: np.ndarray, gm: float) -> np.ndarray:
    L = np.cross(r, v)
    return np.cross(v, L) / gm - r / np.linalg.norm(r)


def semi_major_axis(r: np.ndarray, v: np.ndarray, gm: float) -> float:
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
    sim_id = config_name.split("_")[0]
    sim_label = config_name.split("_", 1)[1].replace("_", " ").replace("+", "+ ")

    fig, ax = dark_fig((7, 7))

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

    all_points = np.vstack([np.array(p)[:, :2] for p in all_positions])
    x_min, y_min = all_points.min(axis=0)
    x_max, y_max = all_points.max(axis=0)
    cx = (x_min + x_max) / 2
    cy = (y_min + y_max) / 2
    hr = max(x_max - x_min, y_max - y_min) / 2 * 1.05
    ax.set_xlim(cx - hr, cx + hr)
    ax.set_ylim(cy - hr, cy + hr)

    ax.set_xlabel("$x$ (AU)", fontsize=10)
    ax.set_ylabel("$y$ (AU)", fontsize=10)
    ax.set_title(
        f"Simulation {sim_id}: {sim_label}  ($t = {T_END:.0f}$ yr)",
        color=TITLE_C,
        fontsize=11,
    )
    ax.set_aspect("equal")
    legend(ax, loc="upper right", fontsize=7)
    savefig(fig, f"trajectories/{sim_id}.png")


def save_energy_error_plot(config_name: str, rows: list[TimeseriesRow]) -> None:
    sim_id = config_name.split("_")[0]
    fig, ax = dark_fig((7, 4))
    times = [r.t for r in rows]
    dE = [r.dE_rel for r in rows]
    ax.semilogy(times, dE, linewidth=1.0, color="#4fa3e0")
    ax.set_xlabel("Time (yr)")
    ax.set_ylabel(r"$\Delta E_{\mathrm{rel}}$")
    ax.set_title(f"Relative Energy Error — {config_name}", color=TITLE_C, fontsize=10)
    ax.set_xlim(0, T_END)
    savefig(fig, f"energy_error/{sim_id}.png")


def plot_energy_error_all(all_rows: dict[str, list[TimeseriesRow]]) -> None:
    fig, ax = dark_fig((10, 5))

    for config_name, rows in all_rows.items():
        times = [r.t for r in rows]
        dE = [r.dE_rel for r in rows]
        colour = CONFIG_COLOURS[config_name]
        label = CONFIG_LABELS[config_name]
        ax.semilogy(times, dE, color=colour, linewidth=0.9, label=label, alpha=0.9)

    ax.set_xlabel("Time (yr)")
    ax.set_ylabel(r"$\Delta E_{\mathrm{rel}}$")
    ax.set_title(
        r"Relative Hamiltonian Error $\Delta E_{\mathrm{rel}}(t)$ — All Configurations",
        color=TITLE_C,
        fontsize=10,
    )
    ax.set_xlim(0, T_END)
    legend(ax, loc="upper left", ncol=2)
    savefig(fig, "energy_error_all.png")


def plot_energy_error_summary(all_rows: dict[str, list[TimeseriesRow]]) -> None:
    """Average energy error bar chart for all 9 configurations."""
    fig, ax = dark_fig((10, 5))

    configs = list(all_rows.keys())
    final_dists = [all_rows[c][-1].dE_rel for c in configs]
    colours = [CONFIG_COLOURS[c] for c in configs]
    labels = [CONFIG_LABELS[c] for c in configs]

    bars = ax.bar(
        range(len(configs)),
        final_dists,
        color=colours,
        edgecolor="#444444",
        linewidth=0.5,
    )

    for bar, val in zip(bars, final_dists):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() * 1.05,
            f"{val:.2e}",
            ha="center",
            va="bottom",
            color=TEXT_C,
            fontsize=6,
            rotation=45,
        )

    ax.set_xticks(range(len(configs)))
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=7)
    ax.set_ylabel(
        r"Final relative energy error $\Delta E_{\mathrm{rel}}(t = 5\ \mathrm{yr})$"
    )
    ax.set_title(
        r"Final Relative Energy Error by Configuration",
        color=TITLE_C,
        fontsize=10,
    )
    ax.set_ylim(bottom=0)
    savefig(fig, "energy_error_summary.png")


def plot_phase_dist_all(all_rows: dict[str, list[TimeseriesRow]]) -> None:
    fig, ax = dark_fig((10, 5))

    ref_rows = all_rows["01_reference"]
    t_ref = [r.t for r in ref_rows]
    d_ref = [r.phase_dist for r in ref_rows]
    ax.fill_between(
        t_ref,
        0,
        d_ref,
        color="#aaaaaa",
        alpha=0.12,
        label="Numerical noise floor (01)",
        zorder=1,
    )
    ax.plot(t_ref, d_ref, color="#aaaaaa", linewidth=0.6, alpha=0.4, zorder=2)

    for config_name, rows in all_rows.items():
        if config_name == "01_reference":
            continue
        times = [r.t for r in rows]
        dists = [r.phase_dist for r in rows]
        colour = CONFIG_COLOURS[config_name]
        label = CONFIG_LABELS[config_name]
        ax.plot(
            times, dists, color=colour, linewidth=1.0, label=label, alpha=0.9, zorder=3
        )

    ax.set_xlabel("Time (yr)")
    ax.set_ylabel(r"Phase-space distance $d(t)$ (AU)")
    ax.set_title(
        r"Phase-Space Distance from Analytical Reference — Simulations 02-09",
        color=TITLE_C,
        fontsize=10,
    )
    ax.set_xlim(0, T_END)
    ax.set_ylim(bottom=0)
    legend(ax, loc="upper left", ncol=2)
    savefig(fig, "phase_dist_all.png")


def plot_phase_dist_summary(all_rows: dict[str, list[TimeseriesRow]]) -> None:
    """Average phase-space distance bar chart for all 9 configurations."""
    fig, ax = dark_fig((10, 5))

    configs = list(all_rows.keys())
    final_dists = [all_rows[c][-1].phase_dist for c in configs]
    colours = [CONFIG_COLOURS[c] for c in configs]
    labels = [CONFIG_LABELS[c] for c in configs]

    bars = ax.bar(
        range(len(configs)),
        final_dists,
        color=colours,
        edgecolor="#444444",
        linewidth=0.5,
    )

    for bar, val in zip(bars, final_dists):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() * 1.05,
            f"{val:.2e}",
            ha="center",
            va="bottom",
            color=TEXT_C,
            fontsize=6,
            rotation=45,
        )

    ax.set_xticks(range(len(configs)))
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=7)
    ax.set_ylabel(r"Final phase-space distance $d(t = 5\ \mathrm{yr})$")
    ax.set_title(
        r"Final Phase-Space Distance by Configuration",
        color=TITLE_C,
        fontsize=10,
    )
    ax.set_ylim(bottom=0)
    savefig(fig, "phase_dist_summary.png")


def plot_eccentricity_all(all_rows: dict[str, list[TimeseriesRow]]) -> None:
    """Eccentricity for selected highlight configs (for the report figure)."""
    fig, ax = dark_fig((10, 5))

    highlight = {"01_reference", "03_+Venus", "05_+Jupiter", "09_full_solar"}

    for config_name, rows in all_rows.items():
        if config_name not in highlight:
            continue
        times = [r.t for r in rows]
        eccs = [r.eccentricity for r in rows]
        colour = CONFIG_COLOURS[config_name]
        label = CONFIG_LABELS[config_name]
        lw = 0.8 if config_name == "01_reference" else 1.4
        ax.plot(times, eccs, color=colour, linewidth=lw, label=label, alpha=0.9)

    ax.set_xlabel("Time (yr)")
    ax.set_ylabel(r"Orbital eccentricity $e(t)$")
    ax.set_title(
        r"Instantaneous Orbital Eccentricity $e(t)$ — Selected Configurations",
        color=TITLE_C,
        fontsize=10,
    )
    ax.set_xlim(0, T_END)
    ax.set_ylim(bottom=0)
    legend(ax, loc="upper left")
    savefig(fig, "eccentricity_all.png")


def plot_eccentricity_summary(all_rows: dict[str, list[TimeseriesRow]]) -> None:
    """Final eccentricity bar chart for all 9 configurations."""
    fig, ax = dark_fig((10, 5))

    configs = list(all_rows.keys())
    final_eccs = [all_rows[c][-1].eccentricity for c in configs]
    colours = [CONFIG_COLOURS[c] for c in configs]
    labels = [CONFIG_LABELS[c] for c in configs]

    bars = ax.bar(
        range(len(configs)),
        final_eccs,
        color=colours,
        edgecolor="#444444",
        linewidth=0.5,
    )

    for bar, val in zip(bars, final_eccs):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() * 1.05,
            f"{val:.2e}",
            ha="center",
            va="bottom",
            color=TEXT_C,
            fontsize=6,
            rotation=45,
        )

    ax.set_xticks(range(len(configs)))
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=7)
    ax.set_ylabel(r"Final eccentricity $e(t = 5\ \mathrm{yr})$")
    ax.set_title(
        r"Final Orbital Eccentricity by Configuration",
        color=TITLE_C,
        fontsize=10,
    )
    ax.set_ylim(bottom=0)
    savefig(fig, "eccentricity_summary.png")


def run_configuration(
    config_name: str,
    body_names: list[str],
) -> tuple[SummaryRow, list[TimeseriesRow]]:
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
    save_energy_error_plot(config_name, rows)

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
    eccs = [r.eccentricity for r in rows]
    dE_rels = [r.dE_rel for r in rows]

    summary = SummaryRow(
        config=config_name,
        n_bodies=n,
        max_phase_dist=max(phase_dists),
        mean_phase_dist=sum(phase_dists) / len(phase_dists),
        max_eccentricity=max(eccs),
        mean_eccentricity=sum(eccs) / len(eccs),
        final_eccentricity=eccs[-1],
        max_dE_rel=max(dE_rels),
    )

    print(
        f"done  |  max d = {summary.max_phase_dist:.3e} AU  "
        f"e_final = {summary.final_eccentricity:.3e}  "
        f"max ΔE = {summary.max_dE_rel:.3e}"
    )

    return summary, rows


def main() -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(os.path.join(RESULTS_DIR, "trajectories"), exist_ok=True)
    os.makedirs(os.path.join(RESULTS_DIR, "energy_error"), exist_ok=True)

    print("\n" + "=" * 65)
    print("  SOLAR SYSTEM PERTURBATION INVESTIGATION")
    print(f"  T_END  = {T_END} yr  |  G = 4π²  (AU³ yr⁻² M☉⁻¹)")
    print(
        f"  a_tol  = {A_TOL:.0e}  |  r_tol = {R_TOL:.0e}  |  dt_max = {DT_MAX:.0e} yr"
    )
    print("=" * 65 + "\n")

    summaries: list[SummaryRow] = []
    all_rows: dict[str, list[TimeseriesRow]] = {}

    for config_name, body_names in CONFIGURATIONS:
        summary, rows = run_configuration(config_name, body_names)
        summaries.append(summary)
        all_rows[config_name] = rows

    # Summary CSV
    summary_path = os.path.join(RESULTS_DIR, "summary.csv")
    with open(summary_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "config",
                "n_bodies",
                "max_phase_dist_AU",
                "mean_phase_dist_AU",
                "max_eccentricity",
                "mean_eccentricity",
                "final_eccentricity",
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
                    f"{s.max_eccentricity:.6e}",
                    f"{s.mean_eccentricity:.6e}",
                    f"{s.final_eccentricity:.6e}",
                    f"{s.max_dE_rel:.6e}",
                ]
            )

    print("\n  Generating summary figures...")
    plot_energy_error_all(all_rows)
    plot_energy_error_summary(all_rows)
    plot_phase_dist_all(all_rows)
    plot_phase_dist_summary(all_rows)
    plot_eccentricity_all(all_rows)
    plot_eccentricity_summary(all_rows)

    print("\n" + "=" * 80)
    print(
        f"  {'Config':<24} {'n':>3}  {'max d (AU)':>12}  "
        f"{'e_final':>12}  {'max ΔE_rel':>12}"
    )
    print("  " + "-" * 72)
    for s in summaries:
        print(
            f"  {s.config:<24} {s.n_bodies:>3}  "
            f"{s.max_phase_dist:>12.3e}  "
            f"{s.final_eccentricity:>12.3e}  "
            f"{s.max_dE_rel:>12.3e}"
        )
    print("=" * 80)
    print(f"\n  Results written to ./{RESULTS_DIR}/\n")


if __name__ == "__main__":
    main()
