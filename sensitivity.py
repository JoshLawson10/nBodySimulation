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
R_TOL: float = 1e-3
DT_MAX: float = 1e-3
DT_INITIAL: float = 1e-4
RESULTS_DIR: str = os.path.join("results", "sensitivity")

TOLERANCE_VALUES: list[float] = [1e-4, 1e-6, 1e-8, 1e-10, 1e-12]
SOFTENING_VALUES: list[float] = [1e-4, 5e-4, 1e-3, 5e-3, 1e-2]

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

TEST_CONFIGS: dict[str, list[str]] = {
    "Jupiter": ["Sun", "Earth", "Jupiter"],
    "Full": [
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
}

BG = "#0d0d0d"
GRID_C = "#2a2a2a"
TEXT_C = "#cccccc"
TITLE_C = "#ffffff"

CONFIG_COLOURS = {
    "Jupiter": "#c88b3a",
    "Full": "#ff6b9d",
}


@dataclass
class SensitivityRow:
    config: str
    param_name: str
    param_value: float
    max_phase_dist: float
    max_dE_rel: float


def reference_state(t: float) -> tuple[np.ndarray, np.ndarray]:
    a = 1.0
    omega = 2 * math.pi
    theta = omega * t
    r = np.array([a * math.cos(theta), a * math.sin(theta), 0.0])
    v = np.array([-a * omega * math.sin(theta), a * omega * math.cos(theta), 0.0])
    return r, v


def make_runner(
    body_names: list[str],
    softening: float,
    a_tol: float,
) -> SimulationRunner:
    bodies = []
    for i, name in enumerate(body_names):
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

    masses = np.array([b.mass for b in bodies], dtype=float)
    system = NBodySystem(masses=masses)
    dynamics = NBodyDynamics(system=system, softening=softening, G=G)
    integrator = AdaptiveRK45Integrator(a_tol=a_tol, r_tol=R_TOL, dt_max=DT_MAX)
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
    ), bodies


def run_trial(
    config_name: str,
    body_names: list[str],
    softening: float,
    a_tol: float,
) -> tuple[float, float]:
    """Returns (max_phase_dist, max_dE_rel)."""
    runner, bodies = make_runner(body_names, softening, a_tol)

    earth_idx = body_names.index("Earth")
    sun_idx = body_names.index("Sun")

    H0_k, H0_u = runner.dynamics.energies(runner.initial_state)
    H0 = H0_k + H0_u

    max_phase_dist = 0.0
    max_dE_rel = 0.0

    for frame in runner.frames(dt=DT_INITIAL, t0=0.0):
        t = frame.time

        r_rel = frame.positions[earth_idx] - frame.positions[sun_idx]
        v_rel = frame.velocities[earth_idx] - frame.velocities[sun_idx]
        r_ref, v_ref = reference_state(t)

        pos_dev = float(np.linalg.norm(r_rel - r_ref))
        vel_dev = float(np.linalg.norm(v_rel - v_ref))
        phase_dist = math.sqrt(pos_dev**2 + vel_dev**2)

        H_now = frame.kinetic_energy + frame.potential_energy
        dE_rel = abs(H_now - H0) / abs(H0) if abs(H0) > 1e-30 else 0.0

        max_phase_dist = max(max_phase_dist, phase_dist)
        max_dE_rel = max(max_dE_rel, dE_rel)

        if t >= T_END:
            break

    return max_phase_dist, max_dE_rel


def dark_fig(figsize: tuple[float, float]):
    fig, axes = plt.subplots(1, 2, figsize=figsize, facecolor=BG)
    for ax in axes:
        ax.set_facecolor(BG)
        ax.tick_params(colors=TEXT_C, labelsize=8)
        ax.xaxis.label.set_color(TEXT_C)
        ax.yaxis.label.set_color(TEXT_C)
        for spine in ax.spines.values():
            spine.set_color("#444444")
        ax.grid(True, color=GRID_C, linewidth=0.5, linestyle="--")
    return fig, axes


def savefig(fig, filename: str) -> None:
    path = os.path.join(RESULTS_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"    saved → {path}")


def legend(ax, **kwargs) -> None:
    ax.legend(
        fontsize=8,
        facecolor="#1a1a1a",
        edgecolor="#444444",
        labelcolor=TEXT_C,
        **kwargs,
    )


def run_tolerance_study() -> list[SensitivityRow]:
    print("\n  Tolerance sensitivity study")
    print("  " + "-" * 55)

    rows: list[SensitivityRow] = []
    SOFTENING = 1e-3

    for config_name, body_names in TEST_CONFIGS.items():
        for a_tol in TOLERANCE_VALUES:
            print(
                f"    [{config_name:<8}]  a_tol = {a_tol:.0e} ... ", end="", flush=True
            )
            max_d, max_dE = run_trial(config_name, body_names, SOFTENING, a_tol)
            print(f"max d = {max_d:.3e}  max ΔE = {max_dE:.3e}")
            rows.append(
                SensitivityRow(
                    config=config_name,
                    param_name="a_tol",
                    param_value=a_tol,
                    max_phase_dist=max_d,
                    max_dE_rel=max_dE,
                )
            )

    return rows


def plot_tolerance(rows: list[SensitivityRow]) -> None:
    fig, axes = dark_fig((12, 5))
    fig.suptitle(
        "Sensitivity to Integrator Tolerance ($a_\\mathrm{tol}$)",
        color=TITLE_C,
        fontsize=11,
    )

    for config_name, colour in CONFIG_COLOURS.items():
        subset = [r for r in rows if r.config == config_name]
        tols = [r.param_value for r in subset]
        dists = [r.max_phase_dist for r in subset]
        dEs = [r.max_dE_rel for r in subset]

        axes[0].loglog(
            tols,
            dists,
            "o-",
            color=colour,
            linewidth=1.2,
            markersize=5,
            label=config_name,
        )
        axes[1].loglog(
            tols,
            dEs,
            "o-",
            color=colour,
            linewidth=1.2,
            markersize=5,
            label=config_name,
        )

    axes[0].set_xlabel("$a_\\mathrm{tol}$")
    axes[0].set_ylabel("$d_\\mathrm{max}$ (AU)")
    axes[0].set_title("Max phase-space distance", color=TITLE_C, fontsize=10)
    axes[0].invert_xaxis()
    legend(axes[0])

    axes[1].set_xlabel("$a_\\mathrm{tol}$")
    axes[1].set_ylabel("$\\Delta E_\\mathrm{rel,max}$")
    axes[1].set_title("Max relative energy error", color=TITLE_C, fontsize=10)
    axes[1].invert_xaxis()
    legend(axes[1])

    plt.tight_layout()
    savefig(fig, "sensitivity_tolerance.png")


def write_tolerance_csv(rows: list[SensitivityRow]) -> None:
    path = os.path.join(RESULTS_DIR, "sensitivity_tolerance.csv")
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "config",
                "a_tol",
                "max_phase_dist_AU",
                "max_dE_rel",
            ]
        )
        for r in rows:
            writer.writerow(
                [
                    r.config,
                    f"{r.param_value:.0e}",
                    f"{r.max_phase_dist:.6e}",
                    f"{r.max_dE_rel:.6e}",
                ]
            )
    print(f"    saved → {path}")


def run_softening_study() -> list[SensitivityRow]:
    print("\n  Softening parameter sensitivity study")
    print("  " + "-" * 55)

    rows: list[SensitivityRow] = []
    A_TOL = 1e-8  # fixed at nominal value

    for config_name, body_names in TEST_CONFIGS.items():
        for softening in SOFTENING_VALUES:
            print(
                f"    [{config_name:<8}]  ε = {softening:.0e} AU ... ",
                end="",
                flush=True,
            )
            max_d, max_dE = run_trial(config_name, body_names, softening, A_TOL)
            print(f"max d = {max_d:.3e}  max ΔE = {max_dE:.3e}")
            rows.append(
                SensitivityRow(
                    config=config_name,
                    param_name="softening",
                    param_value=softening,
                    max_phase_dist=max_d,
                    max_dE_rel=max_dE,
                )
            )

    return rows


def plot_softening(rows: list[SensitivityRow]) -> None:
    fig, axes = dark_fig((12, 5))
    fig.suptitle(
        "Sensitivity to Softening Parameter ($\\epsilon$)",
        color=TITLE_C,
        fontsize=11,
    )

    for config_name, colour in CONFIG_COLOURS.items():
        subset = [r for r in rows if r.config == config_name]
        softs = [r.param_value for r in subset]
        dists = [r.max_phase_dist for r in subset]
        dEs = [r.max_dE_rel for r in subset]

        axes[0].loglog(
            softs,
            dists,
            "o-",
            color=colour,
            linewidth=1.2,
            markersize=5,
            label=config_name,
        )
        axes[1].loglog(
            softs,
            dEs,
            "o-",
            color=colour,
            linewidth=1.2,
            markersize=5,
            label=config_name,
        )

    for ax in axes:
        ax.axvline(
            1e-3,
            color="#ffffff",
            linewidth=0.6,
            linestyle="--",
            alpha=0.4,
            label="Nominal $\\epsilon$",
        )

    axes[0].set_xlabel("$\\epsilon$ (AU)")
    axes[0].set_ylabel("$d_\\mathrm{max}$ (AU)")
    axes[0].set_title("Max phase-space distance", color=TITLE_C, fontsize=10)
    legend(axes[0])

    axes[1].set_xlabel("$\\epsilon$ (AU)")
    axes[1].set_ylabel("$\\Delta E_\\mathrm{rel,max}$")
    axes[1].set_title("Max relative energy error", color=TITLE_C, fontsize=10)
    legend(axes[1])

    plt.tight_layout()
    savefig(fig, "sensitivity_softening.png")


def write_softening_csv(rows: list[SensitivityRow]) -> None:
    path = os.path.join(RESULTS_DIR, "sensitivity_softening.csv")
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "config",
                "softening_AU",
                "max_phase_dist_AU",
                "max_dE_rel",
            ]
        )
        for r in rows:
            writer.writerow(
                [
                    r.config,
                    f"{r.param_value:.0e}",
                    f"{r.max_phase_dist:.6e}",
                    f"{r.max_dE_rel:.6e}",
                ]
            )
    print(f"    saved → {path}")


def main() -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("\n" + "=" * 60)
    print("  SENSITIVITY ANALYSIS")
    print(f"  T_END = {T_END} yr  |  G = 4π²  (AU³ yr⁻² M☉⁻¹)")
    print(f"  Test configurations: {list(TEST_CONFIGS.keys())}")
    print("=" * 60)

    tol_rows = run_tolerance_study()
    write_tolerance_csv(tol_rows)
    plot_tolerance(tol_rows)

    soft_rows = run_softening_study()
    write_softening_csv(soft_rows)
    plot_softening(soft_rows)

    # Print summary tables
    for study_name, rows, param_col in [
        ("TOLERANCE", tol_rows, "a_tol"),
        ("SOFTENING", soft_rows, "softening"),
    ]:
        print(f"\n  {study_name} SUMMARY")
        print(f"  {'Config':<10} {'Param':>10}  {'max d (AU)':>12}  {'max ΔE_rel':>12}")
        print("  " + "-" * 50)
        for r in rows:
            print(
                f"  {r.config:<10} {r.param_value:>10.0e}  "
                f"{r.max_phase_dist:>12.3e}  "
                f"{r.max_dE_rel:>12.3e}"
            )

    print(f"\n  Results written to ./{RESULTS_DIR}/\n")


if __name__ == "__main__":
    main()
