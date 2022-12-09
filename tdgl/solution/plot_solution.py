from typing import Dict, List, Optional, Sequence, Tuple, Union

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from scipy import interpolate

from ..visualization import auto_grid, auto_range_iqr
from .solution import Solution


def setup_color_limits(
    dict_of_arrays: Dict[str, np.ndarray],
    vmin: Union[float, None] = None,
    vmax: Union[float, None] = None,
    share_color_scale: bool = False,
    symmetric_color_scale: bool = False,
    auto_range_cutoff: Optional[Union[float, Tuple[float, float]]] = None,
) -> Dict[str, Tuple[float, float]]:
    """Set up color limits (vmin, vmax) for a dictionary of numpy arrays.

    Args:
        dict_of_arrays: Dict of ``{name: array}`` for which to compute color limits.
        vmin: If provided, this vmin will be used for all arrays. If vmin is not None,
            then vmax must also not be None.
        vmax: If provided, this vmax will be used for all arrays. If vmax is not None,
            then vmin must also not be None.
        share_color_scale: Whether to force all arrays to share the same color scale.
            This option is ignored if vmin and vmax are provided.
        symmetric_color_scale: Whether to use a symmetric color scale (vmin = -vmax).
            This option is ignored if vmin and vmax are provided.
        auto_range_cutoff: Cutoff percentile for :func:`tdgl.solution.plot_solution.auto_range_iqr`.

    Returns:
        A dict of ``{name: (vmin, vmax)}``
    """
    if (vmin is not None and vmax is None) or (vmax is not None and vmin is None):
        raise ValueError("If either vmin or max is provided, both must be provided.")
    if vmin is not None:
        return {name: (vmin, vmax) for name in dict_of_arrays}

    if auto_range_cutoff is None:
        clims = {
            name: (np.nanmin(array), np.nanmax(array))
            for name, array in dict_of_arrays.items()
        }
    else:
        clims = {
            name: auto_range_iqr(array, cutoff_percentile=auto_range_cutoff)
            for name, array in dict_of_arrays.items()
        }

    if share_color_scale:
        # All subplots share the same color scale
        global_vmin = np.inf
        global_vmax = -np.inf
        for vmin, vmax in clims.values():
            global_vmin = min(vmin, global_vmin)
            global_vmax = max(vmax, global_vmax)
        clims = {name: (global_vmin, global_vmax) for name in dict_of_arrays}

    if symmetric_color_scale:
        # Set vmin = -vmax
        new_clims = {}
        for name, (vmin, vmax) in clims.items():
            new_vmax = max(vmax, -vmin)
            new_clims[name] = (-new_vmax, new_vmax)
        clims = new_clims

    return clims


def cross_section(
    dataset_coords: np.ndarray,
    dataset_values: np.ndarray,
    cross_section_coords: Union[np.ndarray, Sequence[np.ndarray]],
    interp_method: str = "linear",
) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
    """Takes a cross-section of the specified dataset values along
    a path given by the given dataset coordinates.

    Args:
        dataset_coords: A shape (n, 2) array of (x, y) coordinates for the dataset.
        dataset_values: A shape (n, ) array of dataset values of which
            to take a cross-section.
        cross_section_coords: A shape (m, 2) array of (x, y) coordinates specifying
            the cross-section path (or a list of such arrays for multiple
            cross sections).
        interp_method: The interpolation method to use: "nearest", "linear", "cubic".

    Returns:
        A list of coordinate arrays, a list of curvilinear coordinate (path) arrays,
        and a list of cross section values.
    """
    valid_methods = ("nearest", "linear", "cubic")
    if interp_method not in valid_methods:
        raise ValueError(
            f"Interpolation method must be one of {valid_methods} "
            f"(got {interp_method})."
        )
    if interp_method == "nearest":
        interpolator = interpolate.NearestNDInterpolator
    elif interp_method == "linear":
        interpolator = interpolate.LinearNDInterpolator
    else:  # "cubic"
        interpolator = interpolate.CloughTocher2DInterpolator

    if not (isinstance(cross_section_coords, Sequence)):
        cross_section_coords = [cross_section_coords]
    cross_section_coords = [np.asarray(c) for c in cross_section_coords]
    for i, arr in enumerate(cross_section_coords):
        if arr.ndim != 2 or arr.shape[1] != 2:
            raise ValueError(
                f"Invalid shape for coordinate array {i}: {arr.shape}. "
                f"Coordinate arrays must have shape (n, 2)."
            )
    # Calculcate curvilinear cross section coordinates
    paths = []
    for c in cross_section_coords:
        path = np.cumsum(np.sqrt(np.sum(np.diff(c, axis=0) ** 2, axis=1)))
        paths.append(np.concatenate([[0], path], axis=0))
    # Calculate cross sections.
    cross_sections = []
    mask = np.isfinite(dataset_values)
    z_interp = interpolator(dataset_coords[mask], dataset_values[mask])
    for c in cross_section_coords:
        cross_sections.append(z_interp(c[:, 0], c[:, 1]))

    return cross_section_coords, paths, cross_sections


def plot_currents(
    solution: Solution,
    ax: Union[plt.Axes, None] = None,
    dataset: Union[str, None] = None,
    units: Union[str, None] = None,
    cmap: str = "inferno",
    colorbar: bool = True,
    auto_range_cutoff: Optional[Union[float, Tuple[float, float]]] = None,
    symmetric_color_scale: bool = False,
    vmin: Union[float, None] = None,
    vmax: Union[float, None] = None,
    streamplot: bool = True,
    min_stream_amp: float = 0.025,
    cross_section_coords: Union[np.ndarray, Sequence[np.ndarray], None] = None,
    **kwargs,
) -> Tuple[plt.Figure, Sequence[plt.Axes]]:
    """Plots the sheet current density for a given :class:`tdgl.Solution`.

    Additional keyword arguments are passed to ``plt.subplots()``.

    .. seealso:

        :meth:`tdgl.Solution.plot_currents`

    Args:
        solution: The Solution from which to extract sheet current.
        dataset: The dataset to plot, either ``"supercurrent"`` or
            ``"normal_current"``. ``None`` indicates the total current density.
        ax: Matplotlib axes on which to plot.
        units: Units in which to plot the current density. Defaults to
            ``solution.current_units / solution.device.length_units``.
        cmap: Name of the matplotlib colormap to use.
        colorbar: Whether to add a colorbar to each subplot.
        auto_range_cutoff: Cutoff percentile for :func:`tdgl.solution.plot_solution.auto_range_iqr`.
        symmetric_color_scale: Whether to use a symmetric color scale (vmin = -vmax).
        vmin: Color scale minimum to use for all layers
        vmax: Color scale maximum to use for all layers
        streamplot: Whether to overlay current streamlines on the plot.
        min_stream_amp: Streamlines will not be drawn anywhere the
            current density is less than min_stream_amp * max(current_density).
            This avoids streamlines being drawn where there is no current flowing.
        cross_section_coords: Shape (m, 2) array of (x, y) coordinates for a
            cross-section (or a list of such arrays).

    Returns:
        matplotlib figure and axes
    """
    device = solution.device
    length_units = device.ureg(device.length_units).units
    old_units = device.ureg(f"{solution.current_units} / {device.length_units}").units
    units = units or old_units
    if isinstance(units, str):
        units = device.ureg(units).units
    if dataset is None:
        J = solution.current_density
    elif dataset in ["supercurrent"]:
        J = solution.supercurrent_density
    elif dataset in ["normal_current"]:
        J = solution.normal_current_density
    else:
        raise ValueError(f"Unexpected dataset: {dataset}.")
    if ax is None:
        fig, ax = plt.subplots(**kwargs)
    else:
        fig = ax.get_figure()

    J = J.to(units).magnitude
    Jx = J[:, 0]
    Jy = J[:, 1]
    Jnorm = np.sqrt(Jx**2 + Jy**2)
    x = solution.device.points[:, 0]
    y = solution.device.points[:, 1]
    t = solution.device.triangles
    clabel = "$|\\,\\vec{K}\\,|$" + f" [${units:~L}$]"
    clim = setup_color_limits(
        {"J": Jnorm},
        vmin=vmin,
        vmax=vmax,
        symmetric_color_scale=symmetric_color_scale,
        auto_range_cutoff=auto_range_cutoff,
    )["J"]
    vmin, vmax = clim
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    im = ax.tripcolor(x, y, t, Jnorm, shading="gouraud", cmap=cmap, norm=norm)
    ax.set_title(f"{clabel.split('[')[0].strip()}")
    ax.set_aspect("equal")
    ax.set_xlabel(f"$x$ [${length_units:~L}$]")
    ax.set_ylabel(f"$y$ [${length_units:~L}$]")
    ax.set_xlim(x.min(), x.max())
    ax.set_ylim(y.min(), y.max())
    if cross_section_coords is not None:
        ax_divider = make_axes_locatable(ax)
        cax = ax_divider.append_axes("bottom", size="40%", pad="30%")
        coords, paths, cross_sections = cross_section(
            np.array([x, y]).T,
            Jnorm,
            cross_section_coords,
        )
        for i, (coord, path, cross) in enumerate(zip(coords, paths, cross_sections)):
            color = f"C{i % 10}"
            cross[~device.contains_points(coord)] = np.nan
            ax.plot(*coord.T, "--", color=color, lw=2)
            ax.plot(*coord[0], "o", color=color)
            ax.plot(*coord[-1], "s", color=color)
            cax.plot(path, cross, color=color, lw=2)
            cax.plot(path[0], cross[0], "o", color=color)
            cax.plot(path[-1], cross[-1], "s", color=color)
        cax.grid(True)
        cax.set_xlabel(f"Distance along cut [${length_units:~L}$]")
        cax.set_ylabel(clabel)
    if streamplot:
        xgrid, ygrid, Jgrid = solution.grid_current_density(
            dataset=dataset,
            grid_shape=200,
            method="cubic",
            units=str(units),
            with_units=False,
        )
        Jx, Jy = Jgrid
        J = np.sqrt(Jx**2 + Jy**2)
        xy = np.array([xgrid.ravel(), ygrid.ravel()]).T
        ix = np.where(~solution.device.contains_points(xy)[0])
        ix = np.unravel_index(ix, J.shape)
        Jx[ix] = np.nan
        Jy[ix] = np.nan
        if min_stream_amp is not None:
            cutoff = np.nanmax(J) * min_stream_amp
            Jx[J < cutoff] = np.nan
            Jy[J < cutoff] = np.nan
        ax.streamplot(xgrid, ygrid, Jx, Jy, color="w", density=1, linewidth=0.75)
    if colorbar:
        cbar = fig.colorbar(im, ax=ax, orientation="vertical")
        cbar.set_label(clabel)
    return fig, ax


def plot_field_at_positions(
    solution: Solution,
    positions: np.ndarray,
    zs: Optional[Union[float, np.ndarray]] = None,
    vector: bool = False,
    units: Union[str, None] = None,
    grid_shape: Union[int, Tuple[int, int]] = (200, 200),
    grid_method: str = "cubic",
    cmap: str = "cividis",
    colorbar: bool = True,
    auto_range_cutoff: Optional[Union[float, Tuple[float, float]]] = None,
    share_color_scale: bool = False,
    symmetric_color_scale: bool = False,
    vmin: Union[float, None] = None,
    vmax: Union[float, None] = None,
    cross_section_coords: Optional[Union[float, List[float]]] = None,
    **kwargs,
) -> Tuple[plt.Figure, Sequence[plt.Axes]]:
    """Plots the Biot-Savart field (either all three components or just the
    z component) at a given set of positions (x, y, z) outside of the device.

    .. note::

        This function plots only the field due to currents flowing in the device.
        It does not include the applied field.

    .. seealso:

        :meth:`tdgl.Solution.plot_field_at_positions`

    Additional keyword arguments are passed to ``plt.subplots()``. This function first
    evaluates the field at ``positions``, then interpolates the resulting fields to a
    rectangular grid for plotting.

    Args:
        solution: The Solution from which to extract fields.
        positions: Shape (m, 2) array of (x, y) coordinates, or (m, 3) array of (x, y, z)
            coordinates at which to calculate the magnetic field.
        zs: z coordinates at which to calculate the field. If positions has shape (m, 3), then
            this argument is not allowed. If zs is a scalar, then the fields are calculated in
            a plane parallel to the x-y plane. If zs is an array, then it must be same length
            as positions.
        vector: Whether to plot the full vector magnetic field or just the z component.
        units: Units in which to plot the fields. Defaults to ``solution.field_units``.
        grid_shape: Shape of the desired rectangular grid. If a single integer ``n``
            is given, then the grid will be square, shape ``(n, n)``.
        grid_method: Interpolation method to use (see :func:`scipy.interpolate.griddata`).
        max_cols: Maximum number of columns in the grid of subplots.
        cmap: Name of the matplotlib colormap to use.
        colorbar: Whether to add a colorbar to each subplot.
        auto_range_cutoff: Cutoff percentile for :func:`tdgl.solution.plot_solution.auto_range_iqr`.
        share_color_scale: Whether to force all layers to use the same color scale.
        symmetric_color_scale: Whether to use a symmetric color scale (vmin = -vmax).
        vmin: Color scale minimum to use for all layers
        vmax: Color scale maximum to use for all layers
        cross_section_coords: Shape (m, 2) array of (x, y) coordinates for a
            cross-section (or a list of such arrays).

    Returns:
        matplotlib figure and axes
    """
    device = solution.device
    # Length units from the Device
    length_units = device.ureg(device.length_units).units
    # The units the fields are currently in
    old_units = device.ureg(solution.field_units).units
    # The units we want to convert to
    if units is None:
        units = old_units
    if isinstance(units, str):
        units = device.ureg(units).units
    fields = solution.field_at_position(
        positions,
        zs=zs,
        vector=vector,
        units=units,
        with_units=False,
    )
    if fields.ndim == 1:
        fields = fields[:, np.newaxis]
    if vector:
        num_subplots = 3
    else:
        num_subplots = 1
    fig, axes = auto_grid(num_subplots, **kwargs)
    if not isinstance(axes, (list, np.ndarray)):
        axes = [axes]
    x, y, *_ = positions.T
    xs = np.linspace(x.min(), x.max(), grid_shape[1])
    ys = np.linspace(y.min(), y.max(), grid_shape[0])
    xgrid, ygrid = np.meshgrid(xs, ys)
    # Shape grid_shape or (grid_shape + (3, ))
    fields = interpolate.griddata(
        positions[:, :2],
        fields,
        (xgrid, ygrid),
        method=grid_method,
    )
    clabels = [f"{label} [${units:~L}$]" for label in ["$H_x$ ", "$H_y$ ", "$H_z$ "]]
    if "[mass]" in units.dimensionality:
        # We want flux density, B = mu0 * H
        clabels = ["$\\mu_0$" + clabel for clabel in clabels]
    if not vector:
        clabels = clabels[-1:]
    fields_dict = {label: fields[:, :, i] for i, label in enumerate(clabels)}
    clim_dict = setup_color_limits(
        fields_dict,
        vmin=vmin,
        vmax=vmax,
        share_color_scale=share_color_scale,
        symmetric_color_scale=symmetric_color_scale,
        auto_range_cutoff=auto_range_cutoff,
    )
    for ax, label in zip(fig.axes, clabels):
        field = fields_dict[label]
        layer_vmin, layer_vmax = clim_dict[label]
        norm = mpl.colors.Normalize(vmin=layer_vmin, vmax=layer_vmax)
        im = ax.pcolormesh(xgrid, ygrid, field, shading="auto", cmap=cmap, norm=norm)
        ax.set_title(f"{label.split('[')[0].strip()}")
        ax.set_aspect("equal")
        ax.set_xlabel(f"$x$ [${length_units:~L}$]")
        ax.set_ylabel(f"$y$ [${length_units:~L}$]")
        ax.set_xlim(xgrid.min(), xgrid.max())
        ax.set_ylim(ygrid.min(), ygrid.max())
        if cross_section_coords is not None:
            ax_divider = make_axes_locatable(ax)
            cax = ax_divider.append_axes("bottom", size="40%", pad="30%")
            coords, paths, cross_sections = cross_section(
                np.array([xgrid.ravel(), ygrid.ravel()]).T,
                field.ravel(),
                cross_section_coords=cross_section_coords,
            )
            for i, (coord, path, cross) in enumerate(
                zip(coords, paths, cross_sections)
            ):
                color = f"C{i % 10}"
                ax.plot(*coord.T, "--", color=color, lw=2)
                ax.plot(*coord[0], "o", color=color)
                ax.plot(*coord[-1], "s", color=color)
                cax.plot(path, cross, color=color, lw=2)
                cax.plot(path[0], cross[0], "o", color=color)
                cax.plot(path[-1], cross[-1], "s", color=color)
            cax.grid(True)
            cax.set_xlabel(f"Distance along cut [${length_units:~L}$]")
            cax.set_ylabel(label)
        if colorbar:
            cbar = fig.colorbar(im, ax=ax, orientation="vertical")
            cbar.set_label(label)
    return fig, axes


def plot_order_parameter(
    solution: Solution,
    squared: bool = False,
    mag_cmap: str = "viridis",
    phase_cmap: str = "twilight_shifted",
    shading: str = "gouraud",
    **kwargs,
) -> Tuple[plt.Figure, Sequence[plt.Axes]]:
    """Plots the magnitude (or the magnitude squared) and
    phase of the complex order parameter, :math:`\\psi=|\\psi|e^{i\\theta}`.

    .. seealso:

        :meth:`tdgl.Solution.plot_order_parameter`

    Args:
        solution: The solution for which to plot the order parameter.
        squared: Whether to plot the magnitude squared, :math:`|\\psi|^2`.
        mag_cmap: Name of the colormap to use for the magnitude.
        phase_cmap: Name of the colormap to use for the phase.
        shading: May be ``"flat"`` or ``"gouraud"``. The latter does some interpolation.

    Returns:
        matplotlib Figure and an array of two Axes objects.
    """
    kwargs.setdefault("figsize", (8, 3))
    kwargs.setdefault("constrained_layout", True)
    device = solution.device
    psi = solution.tdgl_data.psi
    mag = np.abs(psi)
    psi_label = "$|\\psi|$"
    if squared:
        mag = mag**2
        psi_label = "$|\\psi|^2$"
    phase = np.angle(psi) / np.pi
    points = device.points
    triangles = device.triangles
    fig, axes = plt.subplots(1, 2, **kwargs)
    im = axes[0].tripcolor(
        points[:, 0],
        points[:, 1],
        mag,
        triangles=triangles,
        vmin=0,
        vmax=1,
        cmap=mag_cmap,
        shading=shading,
    )
    cbar = fig.colorbar(im, ax=axes[0])
    cbar.set_label(psi_label)
    im = axes[1].tripcolor(
        points[:, 0],
        points[:, 1],
        phase,
        triangles=triangles,
        vmin=-1,
        vmax=1,
        cmap=phase_cmap,
        shading=shading,
    )
    cbar = fig.colorbar(im, ax=axes[1])
    cbar.set_label("$\\theta / \\pi$")
    length_units = device.ureg(device.length_units).units
    for ax in axes:
        ax.set_aspect("equal")
        ax.set_xlabel(f"$x$ [${length_units:~L}$]")
        ax.set_ylabel(f"$y$ [${length_units:~L}$]")
    return fig, axes


def plot_vorticity(
    solution: Solution,
    ax: Union[plt.Axes, None] = None,
    cmap: str = "coolwarm",
    units: Union[str, None] = None,
    auto_range_cutoff: Optional[Union[float, Tuple[float, float]]] = None,
    symmetric_color_scale: bool = True,
    vmin: Union[float, None] = None,
    vmax: Union[float, None] = None,
    shading: str = "gouraud",
    **kwargs,
):
    """Plots the vorticity in the film:
    :math:`\\mathbf{\\omega}=\\mathbf{\\nabla}\\times\\mathbf{K}`.

    .. seealso:

        :meth:`tdgl.Solution.plot_vorticity`

    Args:
        solution: The solution for which to plot the vorticity.
        ax: Matplotlib axes on which to plot.
        cmap: Name of the matplotlib colormap to use.
        units: The units in which to plot the vorticity. Must have dimensions of
            [current] / [length]^2.
        auto_range_cutoff: Cutoff percentile for :func:`tdgl.solution.plot_solution.auto_range_iqr`.
        symmetric_color_scale: Whether to use a symmetric color scale (vmin = -vmax).
        vmin: Color scale minimum.
        vmax: Color scale maximum.
        shading: May be ``"flat"`` or ``"gouraud"``. The latter does some interpolation.

    Returns:
        matplotlib Figure and and Axes.
    """
    if ax is None:
        kwargs.setdefault("constrained_layout", True)
        fig, ax = plt.subplots(**kwargs)
    else:
        fig = ax.get_figure()
    ax.set_aspect("equal")
    device = solution.device
    points = device.points
    triangles = device.triangles
    length_units = device.ureg(device.length_units).units
    if units is None:
        units = solution.vorticity.units
    else:
        units = device.ureg(units)
    v = solution.vorticity.to(units).m
    clim = setup_color_limits(
        {"v": v},
        vmin=vmin,
        vmax=vmax,
        symmetric_color_scale=symmetric_color_scale,
        auto_range_cutoff=auto_range_cutoff,
    )["v"]
    vmin, vmax = clim
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    x, y = points[:, 0], points[:, 1]
    im = ax.tripcolor(
        x,
        y,
        v,
        triangles=triangles,
        cmap=cmap,
        norm=norm,
        shading=shading,
    )
    cbar = fig.colorbar(im, ax=ax)
    ax.set_title("$\\vec{\\omega}=\\vec{\\nabla}\\times\\vec{K}$")
    ax.set_aspect("equal")
    ax.set_xlabel(f"$x$ [${length_units:~L}$]")
    ax.set_ylabel(f"$y$ [${length_units:~L}$]")
    cbar.set_label(f"$\\vec{{\\omega}}\\cdot\\hat{{z}}$ [${units:~L}$]")
    ax.set_xlim(x.min(), x.max())
    ax.set_ylim(y.min(), y.max())
    return fig, ax


def plot_scalar_potential(
    solution: Solution,
    ax: Union[plt.Axes, None] = None,
    cmap: str = "magma",
    auto_range_cutoff: Optional[Union[float, Tuple[float, float]]] = None,
    vmin: Union[float, None] = None,
    vmax: Union[float, None] = None,
    shading: str = "gouraud",
    **kwargs,
):
    """Plots the scalar potential :math:`\\mu(\\mathbf{r})` in the film.

    .. seealso:

        :meth:`tdgl.Solution.plot_scalar_potential`

    Args:
        solution: The solution for which to plot the scalar potential.
        ax: Matplotlib axes on which to plot.
        cmap: Name of the matplotlib colormap to use.
        auto_range_cutoff: Cutoff percentile for :func:`tdgl.solution.plot_solution.auto_range_iqr`.
        vmin: Color scale minimum.
        vmax: Color scale maximum.
        shading: May be ``"flat"`` or ``"gouraud"``. The latter does some interpolation.

    Returns:
        matplotlib Figure and and Axes.
    """
    if ax is None:
        kwargs.setdefault("constrained_layout", True)
        fig, ax = plt.subplots(**kwargs)
    else:
        fig = ax.get_figure()
    ax.set_aspect("equal")
    device = solution.device
    points = device.points
    triangles = device.triangles
    length_units = device.ureg(device.length_units).units
    mu = solution.tdgl_data.mu
    mu = mu - np.nanmin(mu)
    clim = setup_color_limits(
        {"mu": mu},
        vmin=vmin,
        vmax=vmax,
        auto_range_cutoff=auto_range_cutoff,
    )["mu"]
    vmin, vmax = clim
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    x, y = points[:, 0], points[:, 1]
    im = ax.tripcolor(
        x,
        y,
        mu,
        triangles=triangles,
        cmap=cmap,
        norm=norm,
        shading=shading,
    )
    cbar = fig.colorbar(im, ax=ax)
    ax.set_title("$\\mu/v_0$")
    ax.set_aspect("equal")
    ax.set_xlabel(f"$x$ [${length_units:~L}$]")
    ax.set_ylabel(f"$y$ [${length_units:~L}$]")
    cbar.set_label("$\\mu/v_0$")
    ax.set_xlim(x.min(), x.max())
    ax.set_ylim(y.min(), y.max())
    return fig, ax


def _patch_docstring(func):
    other_func = getattr(Solution, func.__name__)
    other_func.__doc__ = (
        other_func.__doc__
        + "\n\n"
        + "\n".join(
            [line for line in func.__doc__.split("\n    ") if "solution:" not in line]
        )
    )
    annotations = func.__annotations__.copy()
    _ = annotations.pop("solution", None)
    other_func.__annotations__.update(annotations)


for func in (
    plot_currents,
    plot_field_at_positions,
    plot_order_parameter,
    plot_scalar_potential,
    plot_vorticity,
):
    _patch_docstring(func)
