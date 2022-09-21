from typing import Optional, Union, List, Dict

import numpy as np

from .device.device import Device


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
        join_style: See :meth:`tdgl.device.components.Polygon.buffer`.
        interp_points: If provided, the resulting polygons will be interpolated to
            ``interp_points`` vertices.

    Returns:
        A dict of ``{hole_name: fluxoid_polygon}``.
    """
    device_polygons = (device.film,) + device.holes
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
