from typing import Dict, List, NamedTuple, Optional, Union

import numpy as np
import pint

from .device.device import Device


class Fluxoid(NamedTuple):
    """The fluxoid for a closed region :math:`S` with boundary :math:`\\partial S`
    is defined as:

    .. math::

        \\begin{split}
        \\Phi^f_S &= \\Phi^f_{S,\\text{ flux}} + \\Phi^f_{S,\\text{ supercurrent}}
        \\\\&=\\int_S \\mu_0 H_z(\\mathbf{r})\\,\\mathrm{d}^2r +
            \\oint_{\\partial S}
            \\mu_0\\Lambda(\\mathbf{r})\\mathbf{K}_s(\\mathbf{r})\\cdot\\mathrm{d}\\mathbf{r}
        \\\\&=\\oint_{\\partial S} \\mathbf{A}(\\mathbf{r})\\cdot\\mathrm{d}\\mathbf{r} +
            \\oint_{\\partial S}
            \\mu_0\\Lambda(\\mathbf{r})\\mathbf{K}_s(\\mathbf{r})\\cdot\\mathrm{d}\\mathbf{r}
        \\end{split}

    Args:
        flux_part: :math:`\\int_S \\mu_0 H_z(\\mathbf{r})\\,\\mathrm{d}^2r=\\oint_{\\partial S}\\mathbf{A}(\\mathbf{r})\\cdot\\mathrm{d}\\mathbf{r}`.
        supercurrent_part: :math:`\\oint_{\\partial S}\\mu_0\\Lambda(\\mathbf{r})\\mathbf{K}_s(\\mathbf{r})\\cdot\\mathrm{d}\\mathbf{r}`.
    """

    flux_part: Union[float, pint.Quantity]
    supercurrent_part: Union[float, pint.Quantity]


def make_fluxoid_polygons(
    device: Device,
    holes: Optional[Union[List[str], str]] = None,
    join_style: str = "mitre",
    interp_points: Optional[int] = None,
) -> Dict[str, np.ndarray]:
    """Generates polygons enclosing the given holes to calculate the fluxoid.

    Args:
        device: The Device for which to generate polygons.
        holes: Name(s) of the hole(s) in the device for which to generate polygons.
            Defaults to all holes in the device.
        join_style: See :meth:`tdgl.Polygon.buffer`.
        interp_points: If provided, the resulting polygons will be interpolated to
            ``interp_points`` vertices.

    Returns:
        A dict of ``{hole_name: fluxoid_polygon}``.
    """
    device_polygons = [device.film] + device.holes
    device_holes = {hole.name: hole for hole in device.holes}
    if holes is None:
        holes = list(device_holes)
    if isinstance(holes, str):
        holes = [holes]
    polygons = {}
    for name in holes:
        hole = device_holes[name]
        hole_poly = hole.polygon
        min_dist = min(
            hole_poly.exterior.distance(other.polygon.exterior)
            for other in device_polygons
            if other.name != name
        )
        delta = min_dist / 2
        new_poly = hole.buffer(delta, join_style=join_style)
        if interp_points:
            new_poly = new_poly.resample(interp_points)
        polygons[name] = new_poly.points
    return polygons
