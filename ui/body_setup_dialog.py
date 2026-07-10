import math
import random
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

from data_types import Body, Vector3


class BodySetupDialog:
    def __init__(self, master):
        self.master = master
        self.result = None
        self.body_frames = []
        self.selected_preset = None

        master.title("N-Body Simulation Setup")
        master.geometry("1000x750")
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

        figure8_btn = Button(
            button_frame,
            text="FIGURE-8",
            command=self.load_figure8,
            bg="#4d0a4d",
            fg="#ff00ff",
            font=("monospace", 10, "bold"),
            padx=10,
            pady=5,
        )
        figure8_btn.pack(side=LEFT, padx=5)

        orrery_btn = Button(
            button_frame,
            text="ORRERY",
            command=self.load_orrery,
            bg="#0a2d4d",
            fg="#00ccff",
            font=("monospace", 10, "bold"),
            padx=10,
            pady=5,
        )
        orrery_btn.pack(side=LEFT, padx=5)

        solar_btn = Button(
            button_frame,
            text="SOLAR SYSTEM",
            command=self.load_solar_system,
            bg="#1b3d0a",
            fg="#99ff66",
            font=("monospace", 10, "bold"),
            padx=10,
            pady=5,
        )
        solar_btn.pack(side=LEFT, padx=5)

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

    def _clear_and_resize(self, n: int) -> None:
        while self.body_frames:
            self.remove_body_row()
        for _ in range(n):
            self.add_body_row()

    def _set_body(
        self,
        idx: int,
        mass: float,
        px: float,
        py: float,
        pz: float,
        vx: float,
        vy: float,
        vz: float,
    ) -> None:
        bf = self.body_frames[idx]
        bf["mass_var"].set(str(mass))
        bf["pos_x"].set(str(px))
        bf["pos_y"].set(str(py))
        bf["pos_z"].set(str(pz))
        bf["vel_x"].set(str(vx))
        bf["vel_y"].set(str(vy))
        bf["vel_z"].set(str(vz))

    def load_figure8(self):
        self._clear_and_resize(3)

        positions = [
            (-0.970004536, 0.243087530, 0),
            (0.0, 0.0, 0),
            (0.970004536, -0.243087530, 0),
        ]
        velocities = [
            (0.466203685 * 2 * math.pi, 0.432365730 * 2 * math.pi, 0),
            (-0.932407370 * 2 * math.pi, -0.864731460 * 2 * math.pi, 0),
            (0.466203685 * 2 * math.pi, 0.432365730 * 2 * math.pi, 0),
        ]

        scale = 5.0
        velocity_scale = 1.0 / np.sqrt(scale)

        for i in range(3):
            self._set_body(
                i,
                mass=1.0,
                px=positions[i][0] * scale,
                py=positions[i][1] * scale,
                pz=positions[i][2] * scale,
                vx=velocities[i][0] * velocity_scale,
                vy=velocities[i][1] * velocity_scale,
                vz=velocities[i][2] * velocity_scale,
            )

        self.selected_preset = "figure8"

    def load_orrery(self):
        self._clear_and_resize(3)

        self._set_body(0, mass=1.0, px=0.0, py=0.0, pz=0.0, vx=0.0, vy=0.0, vz=0.0)
        self._set_body(
            1,
            mass=3.0034896149156e-6,
            px=0.99996874,
            py=0.0,
            pz=0.0,
            vx=0.0,
            vy=6.280538,
            vz=0.0,
        )
        self._set_body(
            2,
            mass=3.6943033497651e-8,
            px=1.00253829,
            py=0.0,
            pz=0.0,
            vx=0.0,
            vy=6.495682,
            vz=0.0,
        )

        self.selected_preset = "orrery"

    def load_solar_system(self):
        self._clear_and_resize(9)

        planets = [
            (1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),  # Sun
            (1.66012e-7, 0.387098, 0.0, 0.0, 0.0, 10.084, 0.0),  # Mercury
            (2.44784e-6, 0.723332, 0.0, 0.0, 0.0, 7.386, 0.0),  # Venus
            (3.00349e-6, 1.000000, 0.0, 0.0, 0.0, 6.283185307, 0.0),  # Earth
            (3.22715e-7, 1.523679, 0.0, 0.0, 0.0, 5.089, 0.0),  # Mars
            (9.54792e-4, 5.2044, 0.0, 0.0, 0.0, 2.755, 0.0),  # Jupiter
            (2.85886e-4, 9.5826, 0.0, 0.0, 0.0, 2.029, 0.0),  # Saturn
            (4.36624e-5, 19.2184, 0.0, 0.0, 0.0, 1.433, 0.0),  # Uranus
            (5.15139e-5, 30.1104, 0.0, 0.0, 0.0, 1.144, 0.0),  # Neptune
        ]

        for i, body in enumerate(planets):
            self._set_body(i, *body)

        self.selected_preset = "solar_system"

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

                color = f"#{random.randint(0, 0xFFFFFF):06x}"
                bodies.append(
                    Body(
                        id=i,
                        mass=mass,
                        position=Vector3(px, py, pz),
                        velocity=Vector3(vx, vy, vz),
                        color=color,
                    )
                )

            self.result = (bodies, self.selected_preset)
            self.master.destroy()
        except ValueError as e:
            messagebox.showerror(
                "Error",
                f"Invalid input format: {e}\nMass and velocity values must be numbers",
            )


def show_body_setup_dialog() -> tuple[list[Body], str | None]:
    root = Tk()
    dialog = BodySetupDialog(root)
    root.wait_window()
    if dialog.result:
        return dialog.result
    return [], None
