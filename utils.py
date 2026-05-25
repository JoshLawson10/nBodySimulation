import numpy as np

from data_types import Body


def pack(bodies: list[Body]) -> np.ndarray:
    r = np.array([b.position.to_array() for b in bodies])
    q = np.array([b.velocity.to_array() for b in bodies])
    return np.concatenate([r.flatten(), q.flatten()])


def unpack(y: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
    r = y[: 3 * n].reshape(n, 3)
    q = y[3 * n :].reshape(n, 3)
    return r, q
