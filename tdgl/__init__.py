from .about import version_dict, version_table
from .device.device import Device
from .device.layer import Layer
from .device.meshing import generate_mesh
from .device.polygon import Polygon
from .em import ureg
from .fluxoid import Fluxoid, make_fluxoid_polygons
from .parameter import Constant, Parameter
from .solution.data import get_current_through_paths
from .solution.plot_solution import (
    plot_current_through_paths,
    plot_currents,
    plot_field_at_positions,
    plot_order_parameter,
    plot_scalar_potential,
    plot_vorticity,
)
from .solution.solution import BiotSavartField, Solution
from .solver.options import SolverOptions
from .solver.solve import solve
from .solver.solver import SolverResult, TDGLSolver
from .version import __git_revision__, __version__
from .visualization.common import non_gui_backend
