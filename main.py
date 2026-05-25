import numpy as np
from matplotlib import pyplot as plt

from data_types import Body, Vector3

N_BODIES = 10
G = 6.674e-11


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


def pack(bodies: list[Body]) -> np.ndarray:
    r = np.array([b.position.to_array() for b in bodies])
    q = np.array([b.velocity.to_array() for b in bodies])
    return np.concatenate([r.flatten(), q.flatten()])


def unpack(y: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
    r = y[: 3 * n].reshape(n, 3)
    q = y[3 * n :].reshape(n, 3)
    return r, q


ax1 = plt.figure().add_subplot(projection="3d")
ax1.set_title("3D Trajectories")
ax1.set_xlabel("x (m)")
ax1.set_ylabel("y (m)")
ax1.set_zlabel("z (m)")
ax1.legend(fontsize=7)

bodies = make_random_bodies(N_BODIES)
masses = np.array([b.mass for b in bodies])
n = len(bodies)
y0 = pack(bodies)

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
