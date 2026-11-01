import numpy as np
import pytest

import tdgl


@pytest.fixture(scope="module")
def strip_device():
    from tdgl.geometry import box

    layer = tdgl.Layer(coherence_length=1, london_lambda=2, thickness=0.1, gamma=1)
    device = tdgl.Device(
        "strip",
        layer=layer,
        film=tdgl.Polygon("film", points=box(8, 24)),
        terminals=[
            tdgl.Polygon("source", points=box(9.6, 0.5, center=(0, 12))),
            tdgl.Polygon("drain", points=box(9.6, 0.5, center=(0, -12))),
        ],
        probe_points=[(0, -7), (0, 7)],
    )
    device.make_mesh(max_edge_length=0.5, smooth=50)
    return device


def test_implicit_laplacian_same_static_state(strip_device):
    # A pinned strip carrying a small current relaxes to a static state (mu psi = 0 in the film), where large
    # implicit steps must give the explicit answer; what is left is an O(dt) error of about 1e-5 in the phase from
    # the normal terminals, and an explicit run that is still settling there at t = 150.
    def relax(**opts):
        options = tdgl.SolverOptions(solve_time=150, save_every=10**6, **opts)
        eps = lambda r: (
            -1.0 if (abs(r[0]) <= 1 and abs(r[1]) <= 1) else 1.0
        )  # noqa: E731
        return tdgl.solve(
            strip_device,
            options,
            disorder_epsilon=eps,
            terminal_currents=dict(source=0.2, drain=-0.2),
        )

    ref = relax()
    new = relax(implicit_laplacian=True, dt_init=1e-4, dt_max=1.0)
    assert len(new.dynamics.dt) < len(ref.dynamics.dt) / 10
    assert abs(ref.dynamics.voltage(0, 1)[-1]) < 1e-5
    # Everywhere but the site next to the source terminal the difference is below 3e-5; that one site
    # carries the O(dt) error of the normal boundary and is about 8e-4 worse.
    assert np.allclose(
        np.abs(new.tdgl_data.psi) ** 2, np.abs(ref.tdgl_data.psi) ** 2, atol=1e-3
    )
    assert np.isclose(
        new.dynamics.phase_difference(0, 1)[-1],
        ref.dynamics.phase_difference(0, 1)[-1],
        atol=3e-5,
    )


def test_implicit_requires_fused_cpu():
    options = tdgl.SolverOptions(
        solve_time=1, implicit_laplacian=True, fused_kernels=False
    )
    with pytest.raises(tdgl.solver.options.SolverOptionsError):
        options.validate()
