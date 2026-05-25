import numpy as np

from data_types import Body


def pack(bodies: list[Body]) -> np.ndarray:
    """Pack body positions and velocities into state vector [r1, r2, ..., rn, v1, v2, ..., vn]."""
    positions = np.array([b.position.to_array() for b in bodies])
    velocities = np.array([b.velocity.to_array() for b in bodies])
    return np.concatenate([positions.flatten(), velocities.flatten()])


def unpack(y: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Unpack state vector into positions (nx3) and velocities (nx3) arrays."""
    positions = y[: 3 * n].reshape(n, 3)
    velocities = y[3 * n :].reshape(n, 3)
    return positions, velocities
