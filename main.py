import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3D

from data_types import Body, Vector3
from utils import pack

"""
Naming Convention:
    position -> q
    displacement -> r
    velocity -> v
    acceleration -> a
"""


N_BODIES = 10


def make_random_bodies(n: int, seed: int | None = 31) -> list[Body]:
    rng = np.random.default_rng(seed)
    cmap = plt.colormaps["tab10"]
    bodies: list[Body] = []

    for i in range(n):
        m = float(rng.uniform(1e20, 1e30))
        r = rng.uniform(-1e11, 1e11, 3)
        q = rng.uniform(-1e4, 1e4, 3)
        color = (
            f"#{int(cmap(i / max(n - 1, 1))[0] * 255):02x}"
            f"{int(cmap(i / max(n - 1, 1))[1] * 255):02x}"
            f"{int(cmap(i / max(n - 1, 1))[2] * 255):02x}"
        )
        bodies.append(
            Body(id=i, mass=m, position=Vector3(*r), velocity=Vector3(*q), color=color)
        )

    return bodies


bodies = make_random_bodies(N_BODIES)
masses = np.array([b.mass for b in bodies])
n = len(bodies)
y0 = pack(bodies)

fig = plt.figure(figsize=(14, 6))
ax1 = fig.add_subplot(121, projection="3d")

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

plt.tight_layout()

for body in bodies:
    ax1.scatter(
        body.position.x,
        body.position.y,
        body.position.z,  # type: ignore
        label=body.name,
        color=body.color,
    )

    ax1.text(
        body.position.x,
        body.position.y,
        body.position.z,
        body.name,
        fontsize=8,
        color=body.color,
    )
plt.show()
