import matplotlib.pyplot as plt
import numpy as np
import pytest

import tdgl

ys = np.linspace(-5, 5, 501)
xs = np.ones_like(ys)
cross_section_coord_params = [
    None,
    np.array([0 * xs, ys]).T,
    np.array([8 * xs, ys]).T,
    [
        np.array([0 * xs, ys]).T,
        np.array([8 * xs, ys]).T,
    ],
]


def test_plot_order_parameter(transport_device_solution: tdgl.Solution):
    fig, axes = transport_device_solution.plot_order_parameter()
    assert isinstance(fig, plt.Figure)
    assert len(axes) == 2


@pytest.mark.parametrize("vmin, vmax", [(None, None), (0, 5), (0, None)])
@pytest.mark.parametrize("streamplot", [False, True])
@pytest.mark.parametrize("cross_section_coords", cross_section_coord_params)
@pytest.mark.parametrize("units", [None, "uA/um", "mA/um"])
@pytest.mark.parametrize("dataset", [None, "supercurrent", "normal_current"])
@pytest.mark.parametrize("colorbar", [False, True])
def test_plot_currents(
    transport_device_solution: tdgl.Solution,
    vmin,
    vmax,
    streamplot,
    cross_section_coords,
    units,
    dataset,
    colorbar,
):
    solution = transport_device_solution
    kwargs = dict(
        vmin=vmin,
        vmax=vmax,
        streamplot=streamplot,
        cross_section_coords=cross_section_coords,
        units=units,
        dataset=dataset,
        colorbar=colorbar,
    )

    if vmax is None and vmin is not None:
        with pytest.raises(ValueError):
            _ = solution.plot_currents(**kwargs)
    else:
        fig, ax = solution.plot_currents(**kwargs)
    plt.close("all")


@pytest.mark.parametrize("vector", [False, True])
@pytest.mark.parametrize(
    "positions, zs",
    [
        (np.random.rand(200).reshape((-1, 2)), np.random.rand(100)),
        (np.random.rand(200).reshape((-1, 2)), 1),
        (np.random.rand(300).reshape((-1, 3)), None),
    ],
)
@pytest.mark.parametrize("units", [None, "mT"])
@pytest.mark.parametrize("auto_range_cutoff", [None, 1])
@pytest.mark.parametrize("cross_section_coords", cross_section_coord_params)
def test_plot_field_at_positions(
    transport_device_solution,
    positions,
    zs,
    units,
    cross_section_coords,
    auto_range_cutoff,
    vector,
):
    solution = transport_device_solution
    fig, ax = solution.plot_field_at_positions(
        positions,
        zs=zs,
        vector=vector,
        grid_shape=(50, 50),
        units=units,
        cross_section_coords=cross_section_coords,
        auto_range_cutoff=auto_range_cutoff,
    )
    plt.close(fig)
