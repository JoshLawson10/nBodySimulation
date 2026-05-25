import numpy as np

from utils import unpack

G = 6.674e-11


def hamiltonian(y: np.ndarray, masses: np.ndarray) -> list[float]:
    """Compute total energy: T (kinetic) + U (potential) = H."""
    n = len(masses)
    positions, velocities = unpack(y, n)

    T = float(np.sum(0.5 * masses * np.sum(velocities**2, axis=1)))

    dr = positions[np.newaxis, :, :] - positions[:, np.newaxis, :]
    distances = np.linalg.norm(dr, axis=2)
    np.fill_diagonal(distances, np.inf)

    upper_idx = np.triu_indices(n, k=1)
    U = float(
        -np.sum(G * masses[upper_idx[0]] * masses[upper_idx[1]] / distances[upper_idx])
    )

    return [T, U]
