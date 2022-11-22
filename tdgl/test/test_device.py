import copy
import os
import pickle
import tempfile

import matplotlib.pyplot as plt
import numpy as np
import pytest
import shapely

import tdgl
from tdgl.solution.plot_solution import non_gui_backend


def test_set_polygon_points():
    box = shapely.geometry.box(0, 0, 1, 1).exterior.coords
    hole = shapely.geometry.box(0.25, 0.25, 0.75, 0.75, ccw=False)
    polygon = shapely.geometry.polygon.Polygon(box, holes=[hole.exterior.coords])

    with pytest.raises(ValueError):
        _ = tdgl.Polygon("bad", points=polygon)

    invalid = shapely.geometry.polygon.LinearRing(
        [(0, 0), (0, 2), (1, 1), (2, 2), (2, 0), (1, 1), (0, 0)]
    )

    with pytest.raises(ValueError):
        _ = tdgl.Polygon(points=invalid)

    x, y = tdgl.geometry.circle(1).T
    z = np.ones_like(x)
    points = np.stack([x, y, z], axis=1)
    with pytest.raises(ValueError):
        _ = tdgl.Polygon(points=points)


def test_polygon_on_boundary(radius=1):
    points = tdgl.geometry.circle(radius, points=501)
    polygon = tdgl.Polygon(points=points)
    Delta_x, Delta_y = polygon.extents
    assert np.isclose(Delta_x, 2 * radius)
    assert np.isclose(Delta_y, 2 * radius)

    smaller = tdgl.geometry.circle(radius - 0.01)
    bigger = tdgl.geometry.circle(radius + 0.01)
    assert polygon.on_boundary(smaller, radius=0.1).all()
    assert polygon.on_boundary(bigger, radius=0.1).all()
    assert not polygon.on_boundary(smaller, radius=0.001).any()
    assert not polygon.on_boundary(bigger, radius=0.001).any()
    assert polygon.on_boundary(smaller, index=True).dtype is np.dtype(int)


def test_polygon_join():

    square1 = tdgl.Polygon(points=tdgl.geometry.box(1))
    square2 = tdgl.Polygon(points=tdgl.geometry.box(1)).translate(dx=0.5, dy=0.5)
    square3 = tdgl.geometry.box(1, center=(-0.25, 0.25))
    name = "name"
    for items in (
        [square1, square2, square3],
        [square1.points, square2.points, square3],
        [square1.polygon, square2.polygon, square3],
    ):
        _ = tdgl.Polygon.from_union(items, name=name)
        _ = tdgl.Polygon.from_difference(items, name=name)
        _ = tdgl.Polygon.from_intersection(items, name=name)

    assert (
        square1.union(square2, square3).polygon
        == tdgl.Polygon.from_union(items, name=name).polygon
    )

    assert (
        square1.intersection(square2, square3).polygon
        == tdgl.Polygon.from_intersection(items, name=name).polygon
    )

    assert (
        square1.difference(square2, square3).polygon
        == tdgl.Polygon.from_difference(items, name=name).polygon
    )

    square1.layer = square2.layer = None
    with pytest.raises(ValueError):
        _ = tdgl.Polygon.from_difference([square1, square1], name=name)

    with pytest.raises(ValueError):
        _ = square1._join_via(square2, "invalid")

    with pytest.raises(ValueError):
        _ = tdgl.Polygon.from_difference(
            [square1, square2],
            name=name,
            symmetric=True,
        )

    assert square1.resample(False) == square1
    assert square1.resample(None).points.shape == square1.points.shape
    assert square1.resample(71).points.shape != square1.points.shape

    with pytest.raises(ValueError):
        bowtie = [(0, 0), (0, 2), (1, 1), (2, 2), (2, 0), (1, 1), (0, 0)]
        _ = tdgl.Polygon(name="bowtie", points=bowtie)

    for min_points, optimesh_steps in [(None, None), (500, None), (500, 10)]:
        mesh = square1.make_mesh(min_points=min_points, optimesh_steps=optimesh_steps)
        if min_points:
            assert mesh.x.shape[0] > min_points


def test_plot_polygon():
    with non_gui_backend():
        ax = tdgl.Polygon("square1", points=tdgl.geometry.box(1)).plot()
        assert isinstance(ax, plt.Axes)


@pytest.fixture(scope="module")
def device():

    layer = tdgl.Layer(london_lambda=1, coherence_length=1, thickness=0.1, z0=0)
    film = tdgl.Polygon("ring", points=tdgl.geometry.ellipse(3, 2, angle=5))
    hole = tdgl.Polygon("hole", points=tdgl.geometry.circle(1))

    offset_film = film.buffer(1, join_style="mitre", as_polygon=False)
    assert isinstance(offset_film, np.ndarray)
    assert offset_film.shape[0] >= film.points.shape[0]

    offset_poly = film.buffer(1)
    assert isinstance(offset_poly, tdgl.Polygon)
    assert film.name in offset_poly.name

    assert film.contains_points([0, 0])
    assert np.array_equal(
        film.contains_points([[0, 0], [2, 1]], index=True), np.array([0, 1])
    )
    assert np.isclose(film.area, np.pi * 3 * 2, rtol=1e-3)

    abstract_regions = [
        tdgl.Polygon(
            "abstract",
            points=tdgl.geometry.box(2.5, angle=45),
        ),
    ]

    device = tdgl.Device(
        "device",
        layer=layer,
        film=film,
        holes=[hole],
        abstract_regions=abstract_regions,
    )

    with pytest.raises(TypeError):
        device.scale(xfact=-1, origin="center")
    with pytest.raises(TypeError):
        device.rotate(90, origin="centroid")

    assert isinstance(device.scale(xfact=-1), tdgl.Device)
    assert isinstance(device.scale(yfact=-1), tdgl.Device)
    assert isinstance(device.rotate(90), tdgl.Device)
    dx = 1
    dy = -1
    dz = 1
    assert isinstance(device.translate(dx, dy, dz=dz), tdgl.Device)
    d = device.copy()
    assert d.translate(dx, dy, dz=dz, inplace=True) is d

    return device


@pytest.fixture(scope="module")
def device_with_mesh():

    with pytest.raises(ValueError):
        _ = tdgl.Polygon("poly", points=tdgl.geometry.circle(1).T)

    layer = tdgl.Layer(london_lambda=1, coherence_length=1, thickness=0.1, z0=0)

    film = tdgl.Polygon(
        "ring",
        points=tdgl.geometry.close_curve(tdgl.geometry.circle(4)),
    )

    holes = [
        tdgl.Polygon("ring_hole", points=tdgl.geometry.circle(2)),
    ]

    device = tdgl.Device(
        "device",
        layer=layer,
        film=film,
        holes=holes,
    )
    assert device.edge_lengths is None
    assert device.triangles is None
    device.make_mesh(min_points=3000)
    assert isinstance(device.edge_lengths, np.ndarray)
    assert isinstance(device.triangles, np.ndarray)
    centroids = tdgl.finite_volume.util.centroids(device.points, device.triangles)
    assert centroids.shape[0] == device.triangles.shape[0]

    print(device)
    assert device == device
    assert device.layer == layer
    assert device.film == film
    assert film != layer
    assert layer != film
    assert layer == layer.copy()
    assert layer is not layer.copy()
    assert film == film.copy()
    assert film is not film.copy()
    assert device != layer

    assert copy.deepcopy(device) == copy.copy(device) == device.copy() == device

    d = device.scale(xfact=-1)
    assert isinstance(d, tdgl.Device)
    assert d.points is None
    d = device.scale(yfact=-1)
    assert isinstance(d, tdgl.Device)
    assert d.points is None
    d = device.rotate(90)
    assert isinstance(d, tdgl.Device)
    assert d.points is None
    dx = 1
    dy = -1
    dz = 1
    assert isinstance(device.translate(dx, dy, dz=dz), tdgl.Device)
    d = device.copy()
    assert d.translate(dx, dy, dz=dz, inplace=True) is d

    for points in ("poly_points", "points"):
        x0, y0 = getattr(device, points).mean(axis=0)
        z0 = device.layer.z0
        with device.translation(dx, dy, dz=dz):
            x, y = getattr(device, points).mean(axis=0)
            assert np.isclose(x, x0 + dx)
            assert np.isclose(y, y0 + dy)
            assert np.isclose(layer.z0, z0 + dz)
        x, y = getattr(device, points).mean(axis=0)
        assert np.isclose(x, x0)
        assert np.isclose(y, y0)
        assert np.isclose(layer.z0, z0)

    return device


@pytest.mark.parametrize("legend", [False, True])
def test_plot_device(device, device_with_mesh, legend, mesh=True):
    with non_gui_backend():
        fig, axes = device.plot(legend=legend)
        with pytest.raises(RuntimeError):
            _ = device.plot(legend=legend, mesh=mesh)
        fig, axes = device_with_mesh.plot(legend=legend, mesh=mesh)
        plt.close("all")


@pytest.mark.parametrize("legend", [False, True])
def test_draw_device(device, legend):
    with non_gui_backend():
        fig, axes = device.draw(exclude="disk", legend=legend)
        fig, axes = device.draw(
            legend=legend,
        )

    with non_gui_backend():
        fig, ax = plt.subplots()
        _ = tdgl.Device(
            "device",
            layer=tdgl.Layer(london_lambda=1, coherence_length=1, thickness=0.1, z0=0),
            film=tdgl.Polygon("disk", points=tdgl.geometry.circle(1)),
        ).draw(ax=ax)
        plt.close("all")


@pytest.mark.parametrize(
    ", ".join(["min_points", "optimesh_steps"]),
    [(None, None), (None, 20), (1200, None), (1200, 20)],
)
def test_make_mesh(device, min_points, optimesh_steps):
    device.make_mesh(
        min_points=min_points,
        optimesh_steps=optimesh_steps,
    )

    assert device.points is not None
    assert device.triangles is not None
    if min_points:
        assert device.points.shape[0] >= min_points


@pytest.mark.parametrize("save_mesh", [False, True])
def test_device_to_file(device, device_with_mesh, save_mesh):

    with tempfile.TemporaryDirectory() as directory:
        path = os.path.join(directory, "device.h5")
        device.to_hdf5(os.path.join(path), save_mesh=save_mesh)
        loaded_device = tdgl.Device.from_hdf5(path)
    assert device == loaded_device

    with tempfile.TemporaryDirectory() as directory:
        path = os.path.join(directory, "device.h5")
        device_with_mesh.to_hdf5(path, save_mesh=save_mesh)
        loaded_device = tdgl.Device.from_hdf5(path)
    assert device_with_mesh == loaded_device


def test_pickle_device(device, device_with_mesh):

    loaded_device = pickle.loads(pickle.dumps(device))
    loaded_device_with_mesh = pickle.loads(pickle.dumps(device_with_mesh))

    assert loaded_device == device
    assert loaded_device_with_mesh == device_with_mesh

    assert loaded_device.ureg("1 m") == loaded_device.ureg("1000 mm")
