from __future__ import annotations

from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np


@dataclass(slots=True, frozen=True)
class BodyStyle:
    color: str
    marker_size: float


class BodyStyleFactory:
    @staticmethod
    def create(
        masses: np.ndarray,
    ) -> list[BodyStyle]:
        cmap = plt.colormaps["hsv"]

        min_mass = float(np.min(masses))
        max_mass = float(np.max(masses))

        styles: list[BodyStyle] = []

        for index, mass in enumerate(masses):
            hue = (index + 0.5) / max(len(masses) - 1, 1)

            rgb = cmap(hue)

            color = (
                f"#{int(rgb[0] * 255):02x}"
                f"{int(rgb[1] * 255):02x}"
                f"{int(rgb[2] * 255):02x}"
            )

            if max_mass > min_mass:
                normalized = (mass - min_mass) / (max_mass - min_mass)

                size = 50.0 + normalized * 450.0
            else:
                size = 250.0

            styles.append(
                BodyStyle(
                    color=color,
                    marker_size=size,
                )
            )

        return styles
