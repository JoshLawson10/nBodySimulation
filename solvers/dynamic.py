from collections.abc import Callable, Generator, Iterable

import numpy as np


def solve_ivp(
    func: Callable,
    y0: np.ndarray,
    dt: float,
    t0: float,
    t_max: float = float("inf"),
    a_tol: float = 1e-8,
    r_tol: float = 1e-3,
    dt_max: float = 2.0,
) -> Generator[tuple[float, np.ndarray], None, None]:
    if isinstance(y0, Iterable):
        y_current = np.array(y0, dtype=float)
        func_ = lambda t, y: np.array(func(t, *y), dtype=float)
    else:
        y_current = y0
        func_ = func

    yield t0, y_current.copy()
    t = t0

    while t < t_max:
        if t_max != float("inf") and t + dt > t_max:
            dt = t_max - t

        k1 = func_(t, y_current)
        k2 = func_(t + dt * 1 / 4, y_current + dt * (k1 * 1 / 4))
        k3 = func_(t + dt * 3 / 8, y_current + dt * (k1 * 3 / 32 + k2 * 9 / 32))
        k4 = func_(
            t + dt * 12 / 13,
            y_current + dt * (k1 * 1932 / 2197 - k2 * 7200 / 2197 + k3 * 7296 / 2197),
        )
        k5 = func_(
            t + dt,
            y_current
            + dt * (k1 * 439 / 216 - k2 * 8 + k3 * 3680 / 513 - k4 * 845 / 4104),
        )
        k6 = func_(
            t + dt * 1 / 2,
            y_current
            + dt
            * (
                -k1 * 8 / 27
                + k2 * 2
                - k3 * 3544 / 2565
                + k4 * 1859 / 4104
                - k5 * 11 / 40
            ),
        )

        k = np.array([k1, k2, k3, k4, k5, k6])

        c4 = np.array([25 / 216, 0, 1408 / 2565, 2197 / 4104, -1 / 5, 0])
        c5 = np.array([16 / 135, 0, 6656 / 12825, 28561 / 56430, -9 / 50, 2 / 55])

        y_next4 = y_current + dt * np.dot(c4, k)
        y_next5 = y_current + dt * np.dot(c5, k)

        error_estimate = np.linalg.norm(y_next5 - y_next4)
        tolerance = a_tol + r_tol * np.linalg.norm(y_next5)

        if error_estimate < tolerance:
            if error_estimate <= 1e-10:
                scale = 2
            else:
                scale = 0.9 * (tolerance / error_estimate) ** 0.2

            t += dt
            y_current = y_next5

            yield t, y_current.copy()

            dt *= np.clip(scale, 0.1, 2.0)
            dt = min(dt, dt_max)
        else:
            scale = 0.9 * (tolerance / error_estimate) ** 0.2
            dt *= float(np.clip(scale, 0.1, 2.0))
            dt = min(dt, dt_max)


def solve_2nd_order_ivp(
    func: Callable,
    y0: np.ndarray,
    v0: np.ndarray,
    dt: float,
    t0: float,
    t_max: float,
    a_tol: float = 1e-8,
    r_tol: float = 1e-3,
    dt_max: float = 2.0,
) -> Generator[tuple[float, np.ndarray], None, None]:
    if isinstance(y0, Iterable) or isinstance(v0, Iterable):
        dim = len(y0) if isinstance(y0, Iterable) else 1
        reduced_func = lambda t, *vars: np.array(
            [*vars[dim:], *func(t, *vars)], dtype=float
        )
    else:
        reduced_func = lambda t, y, yp: np.array([yp, func(t, y, yp)], dtype=float)

    reduced_y0 = np.array([y0, v0], dtype=float).flatten()

    yield from solve_ivp(
        func=reduced_func,
        y0=reduced_y0,
        dt=dt,
        t0=t0,
        t_max=t_max,
        a_tol=a_tol,
        r_tol=r_tol,
        dt_max=dt_max,
    )
