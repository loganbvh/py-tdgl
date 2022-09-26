import os

import numpy as np
import superscreen as sc

from .layers import hypres_squid_layers


def make_polygons():
    coords = {}
    npz_path = os.path.join(os.path.dirname(__file__), "hypres-400nm.npz")
    with np.load(npz_path) as df:
        for name, array in df.items():
            coords[name] = array
    films = ["fc", "fc_shield", "pl", "pl_shield"]
    holes = ["fc_center", "pl_center"]
    abstract_regions = ["bounding_box"]
    sc_polygons = {name: sc.Polygon(name, points=coords[name]) for name in films}
    sc_holes = {name: sc.Polygon(name, points=coords[name]) for name in holes}
    sc_abstract = {
        name: sc.Polygon(name, points=coords[name]) for name in abstract_regions
    }
    return sc_polygons, sc_holes, sc_abstract


def make_squid(align_layers: str = "top"):
    polygons, holes, abstract_regions = make_polygons()
    layers = hypres_squid_layers(align=align_layers)
    layer_mapping = {
        "fc": "BE",
        "fc_center": "BE",
        "fc_shield": "W1",
        "pl": "W1",
        "pl_center": "W1",
        "pl_shield": "W2",
    }
    for name, poly in polygons.items():
        poly.layer = layer_mapping[name]
        poly.points = poly.resample(201)
    for name, poly in holes.items():
        poly.layer = layer_mapping[name]
        poly.points = poly.resample(201)
    bbox = abstract_regions["bounding_box"]
    bounding_box = sc.Polygon("bounding_box", layer="BE", points=bbox)
    return sc.Device(
        "hypres_400nm",
        layers=layers,
        films=list(polygons.values()),
        holes=list(holes.values()),
        abstract_regions=[bounding_box],
        length_units="um",
    )
