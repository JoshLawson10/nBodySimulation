import random

from matplotlib import pyplot as plt

from data_types import Body, Vector3

N_BODIES = 10
G = 6.674e-11


bodies: list[Body] = []

for i in range(N_BODIES):
    mass = random.uniform(1e20, 1e30)
    position = Vector3(
        random.uniform(-1e11, 1e11),
        random.uniform(-1e11, 1e11),
        random.uniform(-1e11, 1e11),
    )
    velocity = Vector3(
        random.uniform(-1e4, 1e4),
        random.uniform(-1e4, 1e4),
        random.uniform(-1e4, 1e4),
    )
    bodies.append(Body(id=i, mass=mass, position=position, velocity=velocity))

ax1 = plt.figure().add_subplot(projection="3d")
ax1.set_title("3D Trajectories")
ax1.set_xlabel("x (m)")
ax1.set_ylabel("y (m)")
ax1.set_zlabel("z (m)")
ax1.legend(fontsize=7)

for body in bodies:
    ax1.scatter(
        body.position.x,
        body.position.y,
        body.position.z,
        label=body.name,
        color=body.colour,
    )

    ax1.text(
        body.position.x,
        body.position.y,
        body.position.z,
        body.name,
        fontsize=8,
        color=body.colour,
    )
plt.show()
