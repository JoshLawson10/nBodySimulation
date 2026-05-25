import numpy as np

from hamiltonian import hamiltonian
from utils import unpack

G = 6.674e-11

C = [0, 1 / 5, 3 / 10, 4 / 5, 8 / 9, 1, 1]

A = [
    [],
    [1 / 5],
    [3 / 40, 9 / 40],
    [44 / 45, -56 / 15, 32 / 9],
    [19372 / 6561, -25360 / 2187, 64448 / 6561, -212 / 729],
    [9017 / 3168, -355 / 33, 46732 / 5247, 49 / 176, -5103 / 18656],
]

B5 = [35 / 384, 0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84, 0]
B4 = [5179 / 57600, 0, 7571 / 16695, 393 / 640, -92097 / 339200, 187 / 2100, 1 / 40]


def derivatives(y: np.ndarray, masses: np.ndarray) -> np.ndarray:
    n = len(masses)
    r, q = unpack(y, n)
    a = np.zeros((n, 3))

    for i in range(n):
        for j in range(n):
            if i is not j:
                r_ij = r[j] - r[i]
                dist = np.linalg.norm(r_ij)
                a[i] += G * masses[j] / dist**3 * r_ij

    return np.concatenate([q.flatten(), a.flatten()])


def rk45_step(
    y: np.ndarray, t: float | np.floating, h: float | np.floating, masses: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    k = []

    for i in range(6):
        y_i = y + h * sum(A[i][j] * k[j] for j in range(len(A[i]))) if A[i] else y
        k.append(derivatives(y_i, masses))

    y5 = y + h * sum(B5[i] * k[i] for i in range(6))
    k.append(derivatives(y5, masses))
    y4 = y + h * sum(B4[i] * k[i] for i in range(7))

    return y5, y4


def solve_ivp_rk45(
    y0: np.ndarray,
    t0: float | np.floating,
    h0: float | np.floating,
    h_min: float | np.floating,
    h_max: float | np.floating,
    masses: np.ndarray,
    t_max: float | np.floating = 1e9,
    tol: float | np.floating = 1e-8,
):
    t = t0
    y = y0.copy()
    h = h0

    safety = 0.9
    max_factor = 5.0
    min_factor = 0.2

    yield t, y.copy(), hamiltonian(y, masses)

    while t < t_max:
        if t + h > t_max:
            h = t_max - t

        y5, y4 = rk45_step(y, t, h, masses)
        error = np.linalg.norm(y5 - y4)

        if error == 0.0:
            factor = max_factor
        else:
            factor = safety * (tol / error) ** 0.2
            factor = min(max_factor, max(min_factor, factor))

        if error <= tol:
            t += h
            y = y5
            yield t, y.copy(), hamiltonian(y, masses)

        h = min(h_max, max(h_min, h * factor))
