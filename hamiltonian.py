import numpy as np

from utils import unpack

G = 6.674e-11


def hamiltonian(y: np.ndarray, masses: np.ndarray) -> list[float]:
    n = len(masses)
    q, v = unpack(y, n)

    T = float(sum(0.5 * masses[i] * np.dot(v[i], v[i]) for i in range(n)))

    U = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.linalg.norm(q[j] - q[i])
            U -= G * masses[i] * masses[j] / dist

    return [T, U]
