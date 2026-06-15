from abc import ABC, abstractmethod
from collections.abc import Generator

import numpy as np
from numpy.typing import NDArray


class Integrator(ABC):
    @abstractmethod
    def solve(
        self,
        func,
        y0: NDArray[np.float64],
        dt: float,
        t0: float,
    ) -> Generator[
        tuple[float, NDArray[np.float64]],
        None,
        None,
    ]:
        raise NotImplementedError
