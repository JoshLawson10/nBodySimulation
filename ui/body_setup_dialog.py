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

    def load_figure8(self):
        print("Loading Figure-8 three-body solution...")
        while len(self.body_frames) > 0:
            self.remove_body_row()

        print(
            "Setting up three equal masses with specific initial conditions for the figure-8 solution..."
        )

        for _ in range(3):
            self.add_body_row()

        print("Assigning initial positions and velocities for the figure-8 solution...")

        positions = [
            (-0.970004536, 0.243087530, 0),
            (0.0, 0.0, 0),
            (0.970004536, -0.243087530, 0),
        ]

        velocities = [
            (0.466203685, 0.432365730, 0),
            (-0.932407370, -0.864731460, 0),
            (0.466203685, 0.432365730, 0),
        ]

        scale = 5.0
        velocity_scale = 1.0 / np.sqrt(scale)

        for i in range(3):
            self.body_frames[i]["mass_var"].set("1.0")
            self.body_frames[i]["pos_x"].set(f"{positions[i][0] * scale:.6f}")
            self.body_frames[i]["pos_y"].set(f"{positions[i][1] * scale:.6f}")
            self.body_frames[i]["pos_z"].set(f"{positions[i][2] * scale:.6f}")
            self.body_frames[i]["vel_x"].set(f"{velocities[i][0] * velocity_scale:.6f}")
            self.body_frames[i]["vel_y"].set(f"{velocities[i][1] * velocity_scale:.6f}")
            self.body_frames[i]["vel_z"].set(f"{velocities[i][2] * velocity_scale:.6f}")

        self.selected_preset = "figure8"

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
