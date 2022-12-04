from typing import Optional, Sequence, Union

import numpy as np
import pint
from scipy import spatial, special
from scipy.constants import mu_0

from .finite_volume.mesh import Mesh

ureg = pint.UnitRegistry()


def convert_field(
    value: Union[np.ndarray, float, str, pint.Quantity],
    new_units: Union[str, pint.Unit],
    old_units: Optional[Union[str, pint.Unit]] = None,
    ureg: Optional[pint.UnitRegistry] = None,
    with_units: bool = True,
) -> Union[pint.Quantity, np.ndarray, float]:
    """Converts a value between different field units, either magnetic field H
    [current] / [length] or flux density B = mu0 * H [mass] / ([curret] [time]^2)).

    Args:
        value: The value to convert. It can either be a numpy array (no units),
            a float (no units), a string like "1 uA/um", or a scalar or array
            ``pint.Quantity``. If value is not a string wiht units or a ``pint.Quantity``,
            then old_units must specify the units of the float or array.
        new_units: The units to convert to.
        old_units: The old units of ``value``. This argument is required if ``value``
            is not a string with units or a ``pint.Quantity``.
        ureg: The ``pint.UnitRegistry`` to use for conversion. If None is given,
            a new instance is created.
        with_units: Whether to return a ``pint.Quantity`` with units attached.

    Returns:
        The converted value, either a pint.Quantity (scalar or array with units),
        or an array or float without units, depending on the ``with_units`` argument.
    """
    if ureg is None:
        ureg = pint.UnitRegistry()
    if isinstance(value, str):
        value = ureg(value)
    if isinstance(value, pint.Quantity):
        old_units = value.units
    if old_units is None:
        raise ValueError(
            "Old units must be specified if value is not a string or pint.Quantity."
        )
    if isinstance(old_units, str):
        old_units = ureg(old_units)
    if isinstance(new_units, str):
        new_units = ureg(new_units)
    if not isinstance(value, pint.Quantity):
        value = value * old_units
    if new_units.dimensionality == old_units.dimensionality:
        value = value.to(new_units)
    elif "[length]" in old_units.dimensionality:
        # value is H in units with dimensionality [current] / [length]
        # and we want B = mu0 * H
        value = (value * ureg("mu0")).to(new_units)
    else:
        # value is B = mu0 * H in units with dimensionality
        # [mass] / ([current] [time]^2) and we want H = B / mu0
        value = (value / ureg("mu0")).to(new_units)
    if not with_units:
        value = value.magnitude
    return value


def biot_savart(
    eval_positions: np.ndarray,
    *,
    current_positions: np.ndarray,
    current_vectors: np.ndarray,
    currents: np.ndarray,
) -> np.ndarray:
    """Calculates the vector magnetic field [Bx, By, Bx] at ``eval_positions``
    due to a discrete set of 1D current elements.

    Input units are meters and Amperes, output units are Tesla.

    Args:
        eval_positions: Shape (n, 3) array of (x, y, z) positions at which to
            evaluate the field.
        current_positions: Shape (m, 3) array of (x, y, z) positions for the
            current elements.
        current_vectors: Shape (m, 3) array of (dx, dy, dy) distance vectors
            indicating the direction and length of the current elements.
        currents: Shape (m, ) or (m, 1) array of current magnitudes for each
            current element.

    Returns:
        Shape (n, 3) pint.Quantity array of the vector magnetic field
        at ``eval_positions``.
    """
    eval_positions = np.atleast_2d(eval_positions)
    current_positions = np.atleast_2d(current_positions)
    current_vectors = np.atleast_2d(current_vectors)
    currents = np.atleast_2d(currents)
    assert eval_positions.shape[-1] == 3
    assert current_positions.shape[-1] == 3
    assert current_vectors.shape[-1] == 3
    assert currents.shape[-1] == 1

    dx = np.subtract.outer(eval_positions[:, 0], current_positions[:, 0])
    dy = np.subtract.outer(eval_positions[:, 1], current_positions[:, 1])
    dz = np.subtract.outer(eval_positions[:, 2], current_positions[:, 2])
    rprime = np.stack([dx, dy, dz], axis=-1)
    denom = (np.linalg.norm(rprime, axis=-1) ** 3)[:, :, np.newaxis]
    integrand = np.cross(current_vectors, rprime) / denom
    integral = np.einsum("jk,ijk -> ik", currents, integrand)
    return mu_0 / (4 * np.pi) * integral * ureg("tesla")


def biot_savart_2d(
    x: Union[float, np.ndarray],
    y: Union[float, np.ndarray],
    z: Union[float, np.ndarray],
    *,
    positions: np.ndarray,
    current_densities: np.ndarray,
    z0: float = 0,
    areas: Optional[np.ndarray] = None,
    length_units: str = "um",
    current_units: str = "uA",
    vector: bool = True,
) -> np.ndarray:
    """Returns the magnetic field (in tesla) from a sheet of current located at
    vertical positon ``z0`` (in units of ``length_units``). The current is
    parameterized by a set of ``current_densities`` (in units of
    ``current_units / length_units``) and x-y ``positions`` (in units of
    ``length_units``), and the field is evaluated at coordinates ``(x, y, z)``.

    .. math::

        \\mu_0H_x(\\vec{r}) &= \\frac{\\mu_0}{4\\pi}\\int_S
        \\frac{J_y(\\vec{r}')(\\vec{r}-\\vec{r}')\\cdot\\hat{z}}
        {|\\vec{r}-\\vec{r}'|^3}\\,\\mathrm{d}^2r'\\\\
        \\mu_0H_y(\\vec{r}) &= \\frac{\\mu_0}{4\\pi}\\int_S
        -\\frac{J_x(\\vec{r}')(\\vec{r}-\\vec{r}')\\cdot\\hat{z}}
        {|\\vec{r}-\\vec{r}'|^3}\\,\\mathrm{d}^2r'\\\\
        \\mu_0H_z(\\vec{r}) &= \\frac{\\mu_0}{4\\pi}\\int_S
        \\frac{J_x(\\vec{r}')(\\vec{r}-\\vec{r}')\\cdot\\hat{y}
        - J_y(\\vec{r}')(\\vec{r}-\\vec{r}')\\cdot\\hat{x}}
        {|\\vec{r}-\\vec{r}'|^3}\\,\\mathrm{d}^2r'


    where :math:`\\vec{r}=(x, y, z)` and :math:`\\vec{r}'=(x', y', z_0)`.

    Args:
        x: x-coordinate(s) at which to evaluate the field.
            Either a scalar or vector with shape ``(n, )``.
        y: y-coordinate(s) at which to evaluate the field.
            Either a scalar or vector with shape ``(n, )``.
        z: z-coordinate(s) at which to evaluate the field. Either a scalar
            or vector with shape ``(n, )``.
        positions: Coordinates ``(x0, y0)`` of the current sheet,
            shape ``(m, 2)``.
        current_densities: 2D current density ``(Jx, Jy)``, shape``(m, 2)``.
        z0: Vertical (z) position of the current sheet.
        areas: Vertex areas for ``positions`` in units of ``length_units**2``. If None,
            the ``positions`` are triangulated to calculate vertex areas.
        length_units: The units for all coordinates.
        current_units: The units for current values. The ``current_densities`` are
            assumed to be in units of ``current_units / length_units``.
        vector: Return the full vector magnetic field (shape ``(n, 3)``) rather
            than just the z-component (shape ``(n, )``).

    Returns:
        Magnetic field in tesla evaluated at ``(x, y, z)``. If ``vector`` is True,
        returns the vector magnetic field :math:`\\mu_0\\vec{H}` (shape ``(n, 3)``).
        Otherwise, returns the the :math:`z`-component, :math:`\\mu_0H_z`
        (shape ``(n,)``).
    """
    # Convert everything to base units: meters and amps / meter.
    to_meter = ureg(length_units).to("m").magnitude
    to_amp_per_meter = ureg(f"{current_units} / {length_units}").to("A / m").magnitude
    x, y, z = np.atleast_1d(x, y, z)
    if z.shape[0] == 1:
        z = z * np.ones_like(x)
    x = x * to_meter
    y = y * to_meter
    z = z * to_meter
    positions, current_densities = np.atleast_2d(positions, current_densities)
    positions = positions * to_meter
    z0 = z0 * to_meter
    current_densities = current_densities * to_amp_per_meter
    # Calculate the pairwise distance between the current sheet and evaluation
    # points for each axis.
    x0, y0 = positions[:, 0], positions[:, 1]
    Jx, Jy = current_densities[:, 0], current_densities[:, 1]
    dx = np.subtract.outer(x, x0)
    dy = np.subtract.outer(y, y0)
    dz = np.subtract.outer(z, z0 * np.ones_like(x0))
    if areas is None:
        # Triangulate the current sheet to assign an effective area to each vertex.
        triangles = spatial.Delaunay(positions).simplices
        mesh = Mesh.from_triangulation(positions[:, 0], positions[:, 1], triangles)
        areas = mesh.areas
    else:
        areas = areas * to_meter**2
    # Evaluate the Biot-Savart integral.
    pref = (mu_0 / (4 * np.pi)) * areas * (dx**2 + dy**2 + dz**2) ** (-3 / 2)
    Jx_dy = np.einsum("ij, ij, j -> i", pref, dy, Jx)
    Jy_dx = np.einsum("ij, ij, j -> i", pref, dx, Jy)
    Bz = Jx_dy - Jy_dx
    if not vector:
        return Bz
    Jy_dz = np.einsum("ij, ij, j -> i", pref, dz, Jy)
    Jx_dz = np.einsum("ij, ij, j -> i", pref, dz, Jx)
    Bx = Jy_dz
    By = -Jx_dz
    return np.array([Bx, By, Bz]).T * ureg("tesla")


def current_loop_vector_potential(
    positions: np.ndarray,
    *,
    loop_center: Sequence[float] = (0, 0, 0),
    loop_radius: float = 1,
    current: float = 1,
    length_units: str = "um",
    current_units: str = "uA",
):
    """Calculates the magnetic vector potential [Ax, Ay] at ``positions``
    due to a 1D current loop.

    Args:
        positions: Shape (n, 3) array of (x, y, z) positions at which to
            evaluate the vector potential.
        loop_center: (x, y, z) coordinates of the current loop center.
        loop_radius: radius of the current loop.
        current: Magnitude of the current flowing in the loop.
        length_units: A string specifying the length units.
        current_units: A string specifying the current units.

    Returns:
        Shape (n, 3) array of the vector potential [Ax, Ay, Az] at ``positions``.
    """
    to_meter = ureg(length_units).to("m").magnitude
    to_amp = ureg(current_units).to("A").magnitude
    # http://www.physics.usu.edu/Wheeler/EMarchive/Jch5Notes.pdf
    positions = np.atleast_2d(positions) * to_meter
    loop_center = np.atleast_2d(loop_center) * to_meter
    a = loop_radius * to_meter
    current = current * to_amp
    positions = positions - loop_center
    # # This is a pint-friendly vector norm.
    # rs = np.sqrt(np.sum(np.square(positions), axis=1))
    rs = np.linalg.norm(positions, axis=1)
    thetas = np.arccos(positions[:, 2] / rs)
    sin_thetas = np.sin(thetas)
    # m == k**2, see docs for scipy.special.ellipk
    denom = rs**2 + a**2 + 2 * a * rs * sin_thetas
    m = 4 * a * rs * sin_thetas / denom
    K = special.ellipk(m)
    E = special.ellipe(m)
    mag = -mu_0 * current * a / (np.pi * m) * (((m - 2) * K + 2 * E)) / np.sqrt(denom)
    # \vec{A} is directed along the azimuthal direction,
    # so here we generate the azimuthal unit vector.
    # Azimuthal angle + pi / 2 to get azimuthal direction.
    phis = np.arctan2(positions[:, 1], positions[:, 0]) + np.pi / 2
    direc = np.array([np.cos(phis), np.sin(phis), np.zeros_like(phis)]).T
    return mag[:, np.newaxis] * direc * ureg("T * m")


def current_loop_field(
    positions: np.ndarray,
    *,
    loop_center: Sequence[float] = (0, 0, 0),
    loop_radius: float = 1e-6,
    current: float = 1e-3,
    num_segments: int = 101,
    length_units: str = "um",
    current_units: str = "uA",
):
    """Calculates the vector magnetic field [Bx, By, Bz] at ``positions``
    due to a 1D current loop, in units of tesla.

    Args:
        positions: Shape (n, 3) array of (x, y, z) positions at which to
            evaluate the vector potential.
        loop_center: (x, y, z) coordinates of the current loop center.
        loop_radius: radius of the current loop.
        current: Magnitude of the current flowing in the loop.
        num_segments: Number of current elements used to model the loop.
        length_units: A string specifying the length units.
        current_units: A string specifying the current units.

    Returns:
        Shape (n, 3) array of the magnetic field [Bx, By, Bz] at ``positions``.
    """
    to_meter = ureg(length_units).to("m").magnitude
    to_amp = ureg(current_units).to("A").magnitude
    positions = np.atleast_2d(positions) * to_meter
    loop_center = np.atleast_2d(loop_center) * to_meter
    loop_radius = loop_radius * to_meter
    current = current * to_amp
    # Create loop positions
    thetas = np.linspace(0, 2 * np.pi, num_segments)
    circ = np.array([np.cos(thetas), np.sin(thetas), np.zeros_like(thetas)]).T
    loop = loop_radius * circ + loop_center
    dloop = np.diff(loop, axis=0)
    loop = loop[:-1]
    return biot_savart(
        positions,
        current_positions=loop,
        current_vectors=dloop,
        currents=current,
    ).to("tesla")


def uniform_Bz_vector_potential(
    positions: np.ndarray,
    Bz: Union[float, str, pint.Quantity],
) -> np.ndarray:
    """Calculates the magnetic vector potential [Ax, Ay, Az] at ``positions``
    due uniform magnetic field along the z-axis with strength ``Bz``.

    Args:
        positions: Shape (n, 3) array of (x, y, z) positions in meters at which to
            evaluate the vector potential.
        Bz: The strength of the uniform field, as a pint-parseable string,
            a pint.Quantity, or a float with units of Tesla.

    Returns:
        Shape (n, 3) array of the vector potential [Ax, Ay, Az] at ``positions``
        in units of Tesla * meter.
    """
    assert isinstance(Bz, (float, str, pint.Quantity)), type(Bz)
    positions = np.atleast_2d(positions)
    assert positions.shape[1] == 3, positions.shape
    if not isinstance(positions, pint.Quantity):
        positions = positions * ureg("meter")
    if isinstance(Bz, str):
        Bz = ureg(Bz)
    if isinstance(Bz, float):
        Bz = Bz * ureg("tesla")
    xs = positions[:, 0]
    ys = positions[:, 1]
    dx = np.ptp(xs)
    dy = np.ptp(ys)
    xs = xs - (xs.min() + dx / 2)
    ys = ys - (ys.min() + dy / 2)
    Ax = -Bz * ys / 2
    Ay = Bz * xs / 2
    A = np.stack([Ax, Ay, np.zeros_like(Ax)], axis=1)
    return A.to("tesla * meter")
