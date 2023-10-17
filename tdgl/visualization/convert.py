from typing import Optional

import h5py
from tqdm import tqdm

from ..solution.solution import Solution
from .common import Quantity
from .io import get_plot_data


def convert_to_xdmf(
    path_to_solution: str,
    xdmf_path: Optional[str] = None,
    dimensionless: bool = False,
) -> None:
    """Convert a :class:`tdgl.Solution` from `HDF5 <https://www.hdfgroup.org/solutions/hdf5/>`_
    to `XDMF <https://www.xdmf.org/index.php/Main_Page>`_.

    XDMF files can be viewed using tools like `ParaView <https://www.paraview.org/>`_.
    This function requires the ``meshio`` Python package.

    Args:
        path_to_solution: Path to the HDF5 file containing the tdgl.Solution
        xdmf_path: Path to the output XDMF file.
            Defaults to ``path_to_solution.replace('.h5', '-converted.xdmf')``
        dimensionless: Save the mesh in dimensionless units
            (scaled to the coherence length).
    """
    try:
        import meshio
    except ImportError as e:
        raise RuntimeError(
            "convert_to_xdmf() requires the meshio Python package."
        ) from e

    solution = Solution.from_hdf5(path_to_solution)
    device = solution.device
    mesh = device.mesh
    points = mesh.sites
    if not dimensionless:
        points = points * device.layer.coherence_length
    cells = [("triangle", mesh.elements)]

    if xdmf_path is None:
        xdmf_path = path_to_solution.replace(".h5", "-converted.xdmf")

    first, last = solution.data_range

    with h5py.File(path_to_solution, "r") as h5file:
        with meshio.xdmf.TimeSeriesWriter(xdmf_path) as writer:
            writer.write_points_cells(points, cells)
            for step in tqdm(range(first, last + 1), desc="Solve steps"):
                solution.solve_step = step
                t = solution.tdgl_data.state["time"]
                point_data = {}
                for name, quantity in Quantity.__members__.items():
                    data, _, _ = get_plot_data(h5file, mesh, quantity, step)
                    point_data[name.lower()] = data
                writer.write_data(t, point_data=point_data)
