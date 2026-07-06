import math
import sys

import numpy as np

from data_types import Body
from integration.rk45 import AdaptiveRK45Integrator
from physics.dynamics import NBodyDynamics
from physics.system import NBodySystem
from simulation.runner import SimulationRunner
from ui.body_setup_dialog import show_body_setup_dialog
from visualisation.single_visualiser import SimulationVisualiser

DEFAULT_DT = 0.01
FIGURE8_DT = 0.005

DEFAULT_STEPS_PER_FRAME = 10
FIGURE8_STEPS_PER_FRAME = 5

HISTORY_CAPACITY = 20_000


def build_initial_state(
    bodies: list[Body],
) -> np.ndarray:
    positions = np.array(
        [[body.position.x, body.position.y, body.position.z] for body in bodies],
        dtype=float,
    )

    velocities = np.array(
        [[body.velocity.x, body.velocity.y, body.velocity.z] for body in bodies],
        dtype=float,
    )

    return np.concatenate([positions.ravel(), velocities.ravel()])


def build_system(
    bodies: list[Body],
) -> NBodySystem:
    masses = np.array([body.mass for body in bodies], dtype=float)

    return NBodySystem(masses=masses)


def print_summary(
    bodies: list[Body],
) -> None:
    """
    Print simulation configuration.
    """
    print(f"\nInitialising simulation with {len(bodies)} bodies")
    print("-" * 60)

    total_mass = 0.0

    for body in bodies:
        total_mass += body.mass

        print(
            f"{body.name:<10} "
            f"m={body.mass:.3g}  "
            f"r=({body.position.x:.2f}, "
            f"{body.position.y:.2f}, "
            f"{body.position.z:.2f})  "
            f"v=({body.velocity.x:.2f}, "
            f"{body.velocity.y:.2f}, "
            f"{body.velocity.z:.2f})"
        )

    print("-" * 60)
    print(f"Total mass: {total_mass:.3g}")


def create_runner(
    bodies: list[Body],
) -> SimulationRunner:
    system = build_system(bodies)

    dynamics = NBodyDynamics(system=system, softening=1e-3, G=4 * math.pi**2)
    integrator = AdaptiveRK45Integrator(a_tol=1e-8, r_tol=1e-3, dt_max=0.001)

    initial_state = build_initial_state(bodies)

    return SimulationRunner(
        dynamics=dynamics,
        integrator=integrator,
        initial_state=initial_state,
        history_capacity=HISTORY_CAPACITY,
    )


def determine_simulation_settings(
    preset: str | None,
) -> tuple[float, int]:
    if preset == "figure8":
        return (FIGURE8_DT, FIGURE8_STEPS_PER_FRAME)

    return (DEFAULT_DT, DEFAULT_STEPS_PER_FRAME)


def main() -> None:
    bodies, preset = show_body_setup_dialog()

    if not bodies:
        print("No bodies configured.")
        sys.exit(1)

    dt, steps_per_frame = determine_simulation_settings(preset)

    print_summary(bodies)

    runner = create_runner(bodies)

    visualizer = SimulationVisualiser(
        runner=runner, dt=dt, steps_per_frame=steps_per_frame, interval_ms=30
    )

    print("\nStarting simulation...")

    visualizer.run()


if __name__ == "__main__":
    main()
