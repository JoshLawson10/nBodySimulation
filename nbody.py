import sys
import time
from collections.abc import Callable, Generator
from dataclasses import dataclass
from tkinter import (
    BOTH,
    LEFT,
    RIGHT,
    Button,
    Entry,
    Frame,
    Label,
    StringVar,
    Tk,
    X,
    messagebox,
)

import numpy as np
from matplotlib import animation, rcParams
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3D

from data_types import Body, Vector3
from solvers.dynamic import solve_ivp

"""
Naming Convention:
    position → r (array of shape (n, 3))
    displacement → Δr
    velocity → v (array of shape (n, 3))
    acceleration → a (array of shape (n, 3))
"""

STEPS_PER_FRAME = 10
HISTORY_CAPACITY = 20000

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
    dt: float = 0.01
    a_tol: float = 1e-8
    r_tol: float = 1e-3
    dt_max: float = 0.1
    steps_per_frame: int = STEPS_PER_FRAME
    interval_ms: int = 30


def get_body_color(index: int, total: int) -> str:
    cmap = plt.colormaps["hsv"]
    hue = index / max(total - 1, 1)
    rgb = cmap(hue)
    return f"#{int(rgb[0] * 255):02x}{int(rgb[1] * 255):02x}{int(rgb[2] * 255):02x}"


class BodySetupDialog:
    def __init__(self, master):
        self.master = master
        self.result = None
        self.body_frames = []

        master.title("N-Body Simulation Setup")
        master.geometry("650x750")
        master.configure(bg="#1a1a1a")

        title = Label(
            master,
            text="N-Body Gravitational Simulation",
            bg="#1a1a1a",
            fg="#00ff99",
            font=("monospace", 14, "bold"),
        )
        title.pack(pady=10)

        info = Label(
            master,
            text="Units: Masses in solar masses (M☉), Positions in AU, Velocities in AU/year",
            bg="#1a1a1a",
            fg="#ffaa00",
            font=("monospace", 9),
        )
        info.pack(pady=5)

        scroll_frame = Frame(master, bg="#1a1a1a")
        scroll_frame.pack(fill=BOTH, expand=True, padx=10, pady=10)

        self.bodies_container = Frame(scroll_frame, bg="#1a1a1a")
        self.bodies_container.pack(fill=BOTH, expand=True)

        button_frame = Frame(master, bg="#1a1a1a")
        button_frame.pack(fill=X, padx=10, pady=10)

        add_btn = Button(
            button_frame,
            text="+ ADD BODY",
            command=self.add_body_row,
            bg="#0a4d2c",
            fg="#00ff99",
            font=("monospace", 10, "bold"),
            padx=10,
            pady=5,
        )
        add_btn.pack(side=LEFT, padx=5)

        remove_btn = Button(
            button_frame,
            text="- REMOVE BODY",
            command=self.remove_body_row,
            bg="#4d0a0a",
            fg="#ff6666",
            font=("monospace", 10, "bold"),
            padx=10,
            pady=5,
        )
        remove_btn.pack(side=LEFT, padx=5)

        random_btn = Button(
            button_frame,
            text="RANDOM",
            command=self.randomize_bodies,
            bg="#4d2c0a",
            fg="#ffaa00",
            font=("monospace", 10, "bold"),
            padx=10,
            pady=5,
        )
        random_btn.pack(side=LEFT, padx=5)

        simulate_btn = Button(
            button_frame,
            text="SIMULATE",
            command=self.simulate,
            bg="#0a4d2c",
            fg="#00ff99",
            font=("monospace", 10, "bold"),
            padx=15,
            pady=5,
        )
        simulate_btn.pack(side=RIGHT, padx=5)

        self.add_body_row()
        self.add_body_row()

    def add_body_row(self):
        frame = Frame(
            self.bodies_container,
            bg="#0d0d0d",
            highlightbackground="#444444",
            highlightthickness=1,
        )
        frame.pack(fill=X, pady=3, padx=0)

        body_num = len(self.body_frames) + 1
        num_label = Label(
            frame,
            text=f"#{body_num}",
            bg="#0d0d0d",
            fg="#00ff99",
            font=("monospace", 9, "bold"),
            width=4,
        )
        num_label.pack(side=LEFT, padx=3)

        mass_label = Label(
            frame,
            text="M☉:",
            bg="#0d0d0d",
            fg="#f0f0f0",
            font=("monospace", 9),
            width=4,
        )
        mass_label.pack(side=LEFT, padx=3)

        mass_var = StringVar(value="1.0")
        mass_entry = Entry(
            frame,
            textvariable=mass_var,
            bg="#1a1a1a",
            fg="#00ccff",
            font=("monospace", 8),
            width=10,
        )
        mass_entry.pack(side=LEFT, padx=2)

        pos_label = Label(
            frame,
            text="POS (AU):",
            bg="#0d0d0d",
            fg="#f0f0f0",
            font=("monospace", 9),
            width=9,
        )
        pos_label.pack(side=LEFT, padx=3)

        pos_x = StringVar(value="0")
        pos_x_entry = Entry(
            frame,
            textvariable=pos_x,
            bg="#1a1a1a",
            fg="#00ccff",
            font=("monospace", 8),
            width=8,
        )
        pos_x_entry.pack(side=LEFT, padx=1)

        pos_y = StringVar(value="0")
        pos_y_entry = Entry(
            frame,
            textvariable=pos_y,
            bg="#1a1a1a",
            fg="#00ccff",
            font=("monospace", 8),
            width=8,
        )
        pos_y_entry.pack(side=LEFT, padx=1)

        pos_z = StringVar(value="0")
        pos_z_entry = Entry(
            frame,
            textvariable=pos_z,
            bg="#1a1a1a",
            fg="#00ccff",
            font=("monospace", 8),
            width=8,
        )
        pos_z_entry.pack(side=LEFT, padx=1)

        vel_label = Label(
            frame,
            text="VEL (AU/yr):",
            bg="#0d0d0d",
            fg="#f0f0f0",
            font=("monospace", 9),
            width=10,
        )
        vel_label.pack(side=LEFT, padx=3)

        vel_x = StringVar(value="0")
        vel_x_entry = Entry(
            frame,
            textvariable=vel_x,
            bg="#1a1a1a",
            fg="#00ccff",
            font=("monospace", 8),
            width=8,
        )
        vel_x_entry.pack(side=LEFT, padx=1)

        vel_y = StringVar(value="0")
        vel_y_entry = Entry(
            frame,
            textvariable=vel_y,
            bg="#1a1a1a",
            fg="#00ccff",
            font=("monospace", 8),
            width=8,
        )
        vel_y_entry.pack(side=LEFT, padx=1)

        vel_z = StringVar(value="0")
        vel_z_entry = Entry(
            frame,
            textvariable=vel_z,
            bg="#1a1a1a",
            fg="#00ccff",
            font=("monospace", 8),
            width=8,
        )
        vel_z_entry.pack(side=LEFT, padx=1)

        self.body_frames.append(
            {
                "frame": frame,
                "mass_var": mass_var,
                "pos_x": pos_x,
                "pos_y": pos_y,
                "pos_z": pos_z,
                "vel_x": vel_x,
                "vel_y": vel_y,
                "vel_z": vel_z,
            }
        )

    def remove_body_row(self):
        if len(self.body_frames) > 0:
            frame_data = self.body_frames.pop()
            frame_data["frame"].destroy()
            for i, bf in enumerate(self.body_frames):
                num_label = bf["frame"].winfo_children()[0]
                num_label.config(text=f"#{i + 1}")

    def randomize_bodies(self):
        rng = np.random.default_rng()
        for body_data in self.body_frames:
            m = float(rng.uniform(0.1, 10.0))
            r = rng.uniform(-10, 10, 3)
            v = rng.uniform(-5, 5, 3)
            body_data["mass_var"].set(f"{m:.3f}")
            body_data["pos_x"].set(f"{r[0]:.2f}")
            body_data["pos_y"].set(f"{r[1]:.2f}")
            body_data["pos_z"].set(f"{r[2]:.2f}")
            body_data["vel_x"].set(f"{v[0]:.2f}")
            body_data["vel_y"].set(f"{v[1]:.2f}")
            body_data["vel_z"].set(f"{v[2]:.2f}")
        self.selected_preset = "random"

    def simulate(self):
        bodies = []
        try:
            for i, body_data in enumerate(self.body_frames):
                mass_str = body_data["mass_var"].get().strip()
                if not mass_str:
                    messagebox.showerror("Error", "All fields must be filled")
                    return

                mass = float(mass_str)
                px = float(body_data["pos_x"].get().strip())
                py = float(body_data["pos_y"].get().strip())
                pz = float(body_data["pos_z"].get().strip())
                vx = float(body_data["vel_x"].get().strip())
                vy = float(body_data["vel_y"].get().strip())
                vz = float(body_data["vel_z"].get().strip())

                color = get_body_color(i, len(self.body_frames))
                bodies.append(
                    Body(
                        id=i,
                        mass=mass,
                        position=Vector3(px, py, pz),
                        velocity=Vector3(vx, vy, vz),
                        color=color,
                    )
                )

            self.result = bodies
            self.master.destroy()
        except ValueError as e:
            messagebox.showerror(
                "Error",
                f"Invalid input format: {e}\nMass and velocity values must be numbers",
            )


def show_body_setup_dialog() -> list[Body]:
    root = Tk()
    dialog = BodySetupDialog(root)
    root.wait_window()
    if dialog.result:
        return dialog.result
    return []


def nbody_dynamics(
    masses: np.ndarray, softening: float = 0.0, G: float = 1.0
) -> Callable:
    n_bodies = len(masses)
    n_coords = n_bodies * 3

    def dynamics(t: float, *state_flat: float) -> np.ndarray:
        state = np.array(state_flat, dtype=float)
        positions = state[:n_coords].reshape(n_bodies, 3)
        velocities = state[n_coords:].reshape(n_bodies, 3)

        accelerations = np.zeros_like(positions)

        for i in range(n_bodies):
            for j in range(i + 1, n_bodies):
                dr = positions[j] - positions[i]
                dist_sq = np.dot(dr, dr) + softening**2
                dist = np.sqrt(dist_sq)
                force_mag = G / (dist_sq * dist)  # G / r^3

                acc_ij = force_mag * dr * masses[j]
                accelerations[i] += acc_ij
                accelerations[j] -= force_mag * dr * masses[i]

        return np.concatenate([velocities.flatten(), accelerations.flatten()])

    return dynamics


def _compute_energies(
    state: np.ndarray,
    masses: np.ndarray,
    softening: float = 0.0,
    G: float = 1.0,
) -> tuple[float, float]:
    n_bodies = len(masses)
    n_coords = n_bodies * 3
    positions = state[:n_coords].reshape(n_bodies, 3)
    velocities = state[n_coords:].reshape(n_bodies, 3)

    kinetic_energy = float(0.5 * np.sum(masses[:, None] * velocities**2))

    potential_energy = 0.0
    for i in range(n_bodies):
        for j in range(i + 1, n_bodies):
            dr = positions[j] - positions[i]
            dist = np.sqrt(np.dot(dr, dr) + softening**2)
            potential_energy -= G * masses[i] * masses[j] / dist

    return kinetic_energy, potential_energy


class NBodySimulator:
    def __init__(
        self,
        bodies: list[Body],
        G: float = 1.0,
        softening: float = 0.001,
    ):
        self.bodies = bodies
        self.masses = np.array([b.mass for b in bodies])
        self.n_bodies = len(self.masses)
        self.G = G
        self.softening = softening

        initial_positions = np.array(
            [[b.position.x, b.position.y, b.position.z] for b in bodies]
        )
        initial_velocities = np.array(
            [[b.velocity.x, b.velocity.y, b.velocity.z] for b in bodies]
        )

        self.initial_state = np.concatenate(
            [initial_positions.flatten(), initial_velocities.flatten()]
        )

        self.dynamics_func = nbody_dynamics(self.masses, self.softening, self.G)

    def simulate(
        self,
        dt: float,
        a_tol: float = 1e-8,
        r_tol: float = 1e-3,
        dt_max: float = 0.1,
        callback: Callable[..., bool] | None = None,
    ) -> Generator[
        tuple[float, np.ndarray, np.ndarray, tuple[float, float]], None, None
    ]:
        n_coords = self.n_bodies * 3

        for t, state in solve_ivp(
            func=self.dynamics_func,
            y0=self.initial_state,
            dt=dt,
            t0=0.0,
            a_tol=a_tol,
            r_tol=r_tol,
            dt_max=dt_max,
        ):
            positions = state[:n_coords].reshape(self.n_bodies, 3)
            velocities = state[n_coords:].reshape(self.n_bodies, 3)
            energies = _compute_energies(state, self.masses, self.softening, self.G)

            if callback is not None:
                should_continue = callback(t, positions, velocities)
                if not should_continue:
                    break

            yield t, positions, velocities, energies


class SimulationVisualizer:
    def __init__(
        self,
        simulator: NBodySimulator,
        config: SimulationConfig,
        body_colors: list[str] | None = None,
        body_sizes: list[float] | None = None,
    ):
        self.simulator = simulator
        self.config = config

        n_coords = simulator.n_bodies * 3
        initial_state = simulator.initial_state
        initial_positions = initial_state[:n_coords].reshape(simulator.n_bodies, 3)

        if body_colors is None:
            self.body_colors = [
                get_body_color(i, simulator.n_bodies) for i in range(simulator.n_bodies)
            ]
        else:
            self.body_colors = body_colors

        if body_sizes is None:
            masses = simulator.masses
            min_mass = masses.min()
            max_mass = masses.max()
            if max_mass > min_mass:
                self.body_sizes = 50 + 450 * (masses - min_mass) / (max_mass - min_mass)
            else:
                self.body_sizes = np.full(simulator.n_bodies, 100)
        else:
            self.body_sizes = body_sizes

        self.ts = np.zeros(HISTORY_CAPACITY)
        self.Hs = np.zeros(HISTORY_CAPACITY)
        self.Ts = np.zeros(HISTORY_CAPACITY)
        self.Us = np.zeros(HISTORY_CAPACITY)
        self.positions_history = np.zeros((simulator.n_bodies, HISTORY_CAPACITY, 3))
        self.frame_count = 0

        self._setup_ui()

    def _setup_ui(self):
        self.fig = plt.figure(figsize=(16, 10))
        self.fig.suptitle(
            "N-BODY GRAVITATIONAL SIMULATION",
            fontsize=16,
            fontweight="bold",
            color="#00ff99",
        )

        self.gs = self.fig.add_gridspec(
            2, 2, hspace=0.3, wspace=0.3, left=0.05, right=0.98, top=0.93, bottom=0.08
        )

        self.ax_3d = self.fig.add_subplot(self.gs[:, 0], projection="3d")
        self.ax_3d.set_facecolor("#1a1a1a")
        self.ax_3d.set_title(
            "3D TRAJECTORIES", fontsize=12, fontweight="bold", pad=10, color="#00ff99"
        )
        self.ax_3d.set_xlabel("X (AU)", fontsize=10, color="#00ccff")
        self.ax_3d.set_ylabel("Y (AU)", fontsize=10, color="#00ccff")
        self.ax_3d.set_zlabel("Z (AU)", fontsize=10, color="#00ccff")
        self.ax_3d.grid(True, alpha=0.15, linestyle="--")

        self.traj_lines = []
        for i in range(self.simulator.n_bodies):
            line = Line3D(
                [],
                [],
                [],
                color=self.body_colors[i],
                linewidth=1.5,
                label=f"Body {i + 1} (m={self.simulator.masses[i]:.3g} M☉)",
                alpha=0.9,
            )
            self.traj_lines.append(line)
            self.ax_3d.add_line(line)

        self.body_markers = []
        for i in range(self.simulator.n_bodies):
            marker = Line3D(
                [],
                [],
                [],
                color=self.body_colors[i],
                marker="o",
                markersize=np.sqrt(self.body_sizes[i]) / 3,
                linestyle="",
                markeredgecolor="#ffffff",
                markeredgewidth=0.5,
            )
            self.body_markers.append(marker)
            self.ax_3d.add_line(marker)

        self.ax_3d.legend(loc="upper left", fontsize=9, framealpha=0.9, fancybox=True)

        self.ax_energy_dh = self.fig.add_subplot(self.gs[0, 1])
        self.ax_energy_dh.set_title(
            "ENERGY DEVIATION", fontsize=11, fontweight="bold", color="#00ff99"
        )
        (self.energy_dh_line,) = self.ax_energy_dh.plot(
            [], [], color="#ff00ff", linewidth=2, label="ΔE"
        )
        self.ax_energy_dh.set_xlabel("Time (years)", fontsize=9)
        self.ax_energy_dh.set_ylabel("ΔE (Energy units)", fontsize=9, color="#ff00ff")
        self.ax_energy_dh.grid(True, alpha=0.15)
        self.ax_energy_dh.tick_params(axis="y", labelcolor="#ff00ff")
        self.ax_energy_dh.axhline(y=0, color="gray", linestyle="--", alpha=0.5)

        self.ax_energy_combined = self.fig.add_subplot(self.gs[1, 1])
        self.ax_energy_combined.set_title(
            "KINETIC & POTENTIAL ENERGY",
            fontsize=11,
            fontweight="bold",
            color="#00ff99",
        )
        (self.kinetic_line,) = self.ax_energy_combined.plot(
            [], [], color="#00ccff", linewidth=2, label="Kinetic (T)"
        )
        (self.potential_line,) = self.ax_energy_combined.plot(
            [], [], color="#ff6600", linewidth=2, label="Potential (U)"
        )
        (self.total_line,) = self.ax_energy_combined.plot(
            [], [], color="#ff00ff", linewidth=2, label="Total (E)"
        )
        self.ax_energy_combined.set_xlabel("Time (years)", fontsize=9)
        self.ax_energy_combined.set_ylabel("Energy (natural units)", fontsize=9)
        self.ax_energy_combined.grid(True, alpha=0.15)
        self.ax_energy_combined.legend(loc="best", fontsize=8)

        self.info_text = self.fig.text(
            0.5,
            0.01,
            "Initialising...",
            ha="center",
            fontsize=10,
            family="monospace",
            bbox={
                "boxstyle": "round",
                "facecolor": "#1a1a1a",
                "edgecolor": "#00ff99",
                "linewidth": 2,
                "pad": 0.8,
            },
            color="#f0f0f0",
        )

    def compute_center_of_mass(self, positions: np.ndarray) -> np.ndarray:
        """Compute center of mass position."""
        total_mass = np.sum(self.simulator.masses)
        return (
            np.sum(positions * self.simulator.masses[:, np.newaxis], axis=0)
            / total_mass
        )

    def run(self):
        """Run the simulation with live visualization."""
        wall_t0 = time.time()
        self.H0 = None

        sim_gen = self.simulator.simulate(
            dt=self.config.dt,
            a_tol=self.config.a_tol,
            r_tol=self.config.r_tol,
            dt_max=self.config.dt_max,
        )

        def update(frame: int):
            """Update function for matplotlib animation."""
            finished = False

            for _ in range(self.config.steps_per_frame):
                try:
                    t, positions, velocities, (T, U) = next(sim_gen)
                except StopIteration:
                    finished = True
                    break

                com = self.compute_center_of_mass(positions)
                positions_rel = positions - com

                if self.H0 is None:
                    self.H0 = T + U
                    print(f"Initial total energy: {self.H0:.6e}")

                if self.frame_count < HISTORY_CAPACITY:
                    self.ts[self.frame_count] = t
                    self.Hs[self.frame_count] = T + U - self.H0
                    self.Ts[self.frame_count] = T
                    self.Us[self.frame_count] = U
                    self.positions_history[:, self.frame_count, :] = positions_rel
                    self.frame_count += 1
                else:
                    finished = True
                    break

            if self.frame_count == 0 or finished:
                if finished:
                    self.info_text.set_text("SIMULATION COMPLETE")
                return (
                    self.traj_lines
                    + self.body_markers
                    + [
                        self.energy_dh_line,
                        self.kinetic_line,
                        self.potential_line,
                        self.total_line,
                    ]
                )

            actual_frames = self.frame_count
            ts_data = self.ts[:actual_frames]
            positions_data = self.positions_history[:, :actual_frames, :]

            for i in range(self.simulator.n_bodies):
                self.traj_lines[i].set_data(
                    positions_data[i, :, 0], positions_data[i, :, 1]
                )
                self.traj_lines[i].set_3d_properties(positions_data[i, :, 2])

                last_idx = min(actual_frames - 1, positions_data.shape[1] - 1)
                self.body_markers[i].set_data(
                    [positions_data[i, last_idx, 0]], [positions_data[i, last_idx, 1]]
                )
                self.body_markers[i].set_3d_properties([positions_data[i, last_idx, 2]])

            if actual_frames > 0:
                visible_positions = positions_data[
                    :, max(0, actual_frames - 500) : actual_frames, :
                ]
                all_coords = visible_positions.reshape(-1, 3)
                if len(all_coords) > 0:
                    center = np.mean(all_coords, axis=0)
                    max_range = np.max(np.ptp(all_coords, axis=0)) / 2
                    margin = max(max_range * 0.1, 0.1)

                    self.ax_3d.set_xlim(
                        center[0] - max_range - margin, center[0] + max_range + margin
                    )
                    self.ax_3d.set_ylim(
                        center[1] - max_range - margin, center[1] + max_range + margin
                    )
                    self.ax_3d.set_zlim(
                        center[2] - max_range - margin, center[2] + max_range + margin
                    )

            self.energy_dh_line.set_data(ts_data, self.Hs[:actual_frames])
            self.kinetic_line.set_data(ts_data, self.Ts[:actual_frames])
            self.potential_line.set_data(ts_data, self.Us[:actual_frames])
            self.total_line.set_data(ts_data, self.Hs[:actual_frames])

            for ax in [self.ax_energy_dh, self.ax_energy_combined]:
                ax.relim()
                ax.autoscale_view()

            wall_elapsed = time.time() - wall_t0
            status = "● COMPLETE" if finished else "● RUNNING"
            speed_ratio = ts_data[-1] / wall_elapsed if wall_elapsed > 0 else 0
            energy_error_pct = (
                abs(self.Hs[actual_frames - 1])
                / (abs(self.H0) if self.H0 != 0 else 1)
                * 100
            )

            status_str = (
                f"{status}  │  "
                f"Time: {ts_data[-1]:.2f} yr  │  "
                f"Real: {wall_elapsed:.1f} s  │  "
                f"Speed: {speed_ratio:.1f}x  │  "
                f"ΔE: {self.Hs[actual_frames - 1]:.2e} ({energy_error_pct:.6f}%)"
            )
            self.info_text.set_text(status_str)

            return (
                self.traj_lines
                + self.body_markers
                + [
                    self.energy_dh_line,
                    self.kinetic_line,
                    self.potential_line,
                    self.total_line,
                ]
            )

        ani = animation.FuncAnimation(
            self.fig,
            update,
            interval=self.config.interval_ms,
            blit=False,
            cache_frame_data=False,
            save_count=0,
        )

        plt.show()


def main():
    bodies = show_body_setup_dialog()

    if not bodies:
        print("No bodies configured. Exiting.")
        sys.exit(1)

    print(f"Initializing simulation with {len(bodies)} bodies...")
    total_mass = 0
    for body in bodies:
        total_mass += body.mass
        print(
            f"  Body {body.id}: m={body.mass:.3g} M☉, "
            f"r=({body.position.x:.2f}, {body.position.y:.2f}, {body.position.z:.2f}) AU, "
            f"v=({body.velocity.x:.2f}, {body.velocity.y:.2f}, {body.velocity.z:.2f}) AU/yr"
        )
    print(f"  Total system mass: {total_mass:.3g} M☉")

    simulator = NBodySimulator(bodies, G=1.0, softening=0.001)

    config = SimulationConfig(
        t0=0.0,
        dt=dt,
        a_tol=1e-8,
        r_tol=1e-3,
        dt_max=0.1,
        steps_per_frame=steps_per_frame,
        interval_ms=30,
    )

    visualizer = SimulationVisualizer(simulator, config)

    print("\nStarting simulation visualization...")
    print("Controls:")
    print("  - Mouse drag: Rotate view")
    print("  - Mouse wheel: Zoom")
    print("  - Close window to exit")
    print("=" * 60)

    visualizer.run()


if __name__ == "__main__":
    main()
