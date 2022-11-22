from typing import Tuple

import numpy as np

from ..em import current_loop_vector_potential
from ..parameter import Parameter


def loop_vector_potential(
    x,
    y,
    z,
    *,
    current: float,
    radius: float,
    center: Tuple[float, float, float] = (0, 0, 0),
    current_units: str = "uA",
    field_units: str = "mT",
    length_units: str = "um",
):
    if z.ndim == 0:
        z = z * np.ones_like(x)
    positions = np.array([x.squeeze(), y.squeeze(), z.squeeze()]).T
    A = current_loop_vector_potential(
        positions,
        loop_center=center,
        loop_radius=radius,
        current=current,
        current_units=current_units,
        length_units=length_units,
    )
    return A.to(f"{field_units} * {length_units}").magnitude


def CurrentLoop(
    *,
    current: float,
    radius: float,
    center: Tuple[float, float, float],
    current_units: str = "uA",
    field_units: str = "mT",
    length_units: str = "um",
) -> Parameter:
    """Returns a Parameter that computes the vector potential due to a 1D loop of current.

    Args:
        current: The current flowing in the loop.
        radius: The loop radius.
        center: The ``(x, y, z)`` position of the loop center.
        current_units: Units for current.
        field_units: Units for magnetic field.
        length_units: Units for length.

    Returns:
        A Parameter that computes the vector potential.
    """
    return Parameter(
        loop_vector_potential,
        current=current,
        radius=radius,
        center=center,
        current_units=current_units,
        field_units=field_units,
        length_units=length_units,
    )
