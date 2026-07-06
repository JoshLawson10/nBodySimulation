import time

import numpy as np
from matplotlib import animation, rcParams
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3D

from simulation.runner import SimulationRunner
from visualisation.styles import BodyStyleFactory

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


class SimulationVisualiser:
    def __init__(
        self,
        runner: SimulationRunner,
        *,
        dt: float,
        interval_ms: int,
        steps_per_frame: int = 1,
    ) -> None:
        self.runner = runner
        self.dt = dt
        self.interval_ms = interval_ms

        self.frame_generator = runner.frames(dt=dt)
        self.styles = BodyStyleFactory.create(runner.dynamics.system.masses)

        self._pace_start = time.perf_counter()
        self.wall_start = time.time()

        self._setup_figure()

    def _setup_figure(self) -> None:
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
        self.ax_energy_combined.set_xlabel("Time (years)", fontsize=9)
        self.ax_energy_combined.set_ylabel("Energy (natural units)", fontsize=9)
        self.ax_energy_combined.grid(True, alpha=0.15)

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

        self._create_artists()

    def _create_artists(self) -> None:
        body_count = self.runner.dynamics.body_count

        self.traj_lines: list[Line3D] = []
        self.body_markers: list[Line3D] = []

        for index in range(body_count):
            style = self.styles[index]

            trajectory = Line3D(
                [],
                [],
                [],
                color=style.color,
                linewidth=1.5,
                label=f"Body {index + 1} (m={self.runner.dynamics.system.masses[index]:.3g} M☉)",
            )
            marker = Line3D(
                [],
                [],
                [],
                marker="o",
                linestyle="",
                color=style.color,
                markersize=np.sqrt(style.marker_size) / 3,
            )

            self.ax_3d.add_line(trajectory)
            self.ax_3d.add_line(marker)
            self.traj_lines.append(trajectory)
            self.body_markers.append(marker)

        (self.kinetic_line,) = self.ax_energy_combined.plot(
            [], [], color="#00ccff", linewidth=2, label="Kinetic (T)"
        )
        (self.potential_line,) = self.ax_energy_combined.plot(
            [], [], color="#ff6600", linewidth=2, label="Potential (U)"
        )
        (self.total_line,) = self.ax_energy_combined.plot(
            [], [], color="#ff00ff", linewidth=2, label="Total (E)"
        )

        self.ax_3d.legend(loc="upper left", fontsize=9, framealpha=0.9, fancybox=True)
        self.ax_energy_combined.legend(loc="best", fontsize=8)

    def _advance_simulation(self) -> bool:
        frame_budget_s = self.interval_ms / 1000.0

        wall_now = time.perf_counter()
        target_sim_t = wall_now - self._pace_start

        while True:
            try:
                frame = next(self.frame_generator)
            except StopIteration:
                return True

            if frame.time >= target_sim_t:
                break

        elapsed_step = time.perf_counter() - wall_now
        sleep_s = frame_budget_s - elapsed_step
        if sleep_s > 0:
            time.sleep(sleep_s)

        return False

    def _draw_trajectories(self) -> None:
        frames = self.runner.history.frames
        if not frames:
            return

        body_count = self.runner.dynamics.body_count
        latest = frames[-1]
        trajectories = np.stack(
            [frame.positions - frame.center_of_mass for frame in frames]
        )

        for body_index in range(body_count):
            coords = trajectories[:, body_index, :]
            self.traj_lines[body_index].set_data_3d(
                coords[:, 0], coords[:, 1], coords[:, 2]
            )
            current = latest.positions[body_index] - latest.center_of_mass
            self.body_markers[body_index].set_data_3d(
                [current[0]], [current[1]], [current[2]]
            )

    def _draw_energy(self) -> None:
        frames = self.runner.history.frames
        if not frames:
            return

        ts = np.array([f.time for f in frames])
        kinetic = np.array([f.kinetic_energy for f in frames])
        potential = np.array([f.potential_energy for f in frames])
        total = np.array([f.total_energy for f in frames])

        self.energy_dh_line.set_data(ts, total - total[0])
        self.kinetic_line.set_data(ts, kinetic)
        self.potential_line.set_data(ts, potential)
        self.total_line.set_data(ts, total)

        self.ax_energy_dh.relim()
        self.ax_energy_dh.autoscale_view()
        self.ax_energy_combined.relim()
        self.ax_energy_combined.autoscale_view()

    def _update_bounds(self) -> None:
        frames = self.runner.history.frames
        if not frames:
            return

        latest = frames[-1]
        positions = latest.positions - latest.center_of_mass
        center = np.mean(positions, axis=0)
        radius = max(np.max(np.ptp(positions, axis=0)) / 2, 0.1)

        self.ax_3d.set_xlim(center[0] - radius, center[0] + radius)
        self.ax_3d.set_ylim(center[1] - radius, center[1] + radius)
        self.ax_3d.set_zlim(center[2] - radius, center[2] + radius)

    def _draw_status(self) -> None:
        if len(self.runner.history) == 0:
            return

        latest = self.runner.history.latest
        elapsed = time.time() - self.wall_start
        speed = latest.time / elapsed if elapsed > 0 else 0
        stats = self.runner.statistics

        self.info_text.set_text(
            f"t={latest.time:.2f} yr | "
            f"speed={speed:.2f}x | "
            f"ΔE={stats.energy_error:.2e} (relative: {stats.energy_error_percent:.6f}%)"
        )

    def _update(self, frame_number: int):
        del frame_number

        self._advance_simulation()
        self._draw_trajectories()
        self._draw_energy()
        self._update_bounds()
        self._draw_status()

        return (
            self.traj_lines
            + self.body_markers
            + [self.kinetic_line, self.potential_line, self.total_line]
        )

    def run(self) -> None:
        self._anim = animation.FuncAnimation(
            self.fig,
            self._update,
            interval=self.interval_ms,
            cache_frame_data=False,
            blit=False,
        )
        plt.show()
