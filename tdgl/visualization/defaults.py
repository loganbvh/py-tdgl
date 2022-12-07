from dataclasses import dataclass
from enum import Enum
from typing import Sequence


class Quantity(Enum):
    ORDER_PARAMETER = "Order parameter"
    PHASE = "Phase"
    SUPERCURRENT = "Supercurrent density"
    NORMAL_CURRENT = "Normal current density"
    VORTICITY = "Vorticity"
    SCALAR_POTENTIAL = "Scalar potential"
    APPLIED_VECTOR_POTENTIAL = "Applied vector potential"
    INDUCED_VECTOR_POTENTIAL = "Induced vector potential"
    ALPHA = "Alpha"

    @classmethod
    def get_keys(cls) -> Sequence[str]:
        return list(item.name for item in Quantity)

    @classmethod
    def from_key(cls, key: str) -> "Quantity":
        return Quantity[key.upper()]


colormaps = {
    Quantity.ORDER_PARAMETER: "viridis",
    Quantity.PHASE: "twilight_shifted",
    Quantity.SUPERCURRENT: "inferno",
    Quantity.NORMAL_CURRENT: "inferno",
    Quantity.SCALAR_POTENTIAL: "magma",
    Quantity.APPLIED_VECTOR_POTENTIAL: "cividis",
    Quantity.INDUCED_VECTOR_POTENTIAL: "cividis",
    Quantity.ALPHA: "viridis",
    Quantity.VORTICITY: "coolwarm",
}


@dataclass
class PlotDefault:
    cmap: str
    clabel: str
    xlabel: str = "$x/\\xi$"
    ylabel: str = "$y/\\xi$"


PLOT_DEFAULTS = {
    Quantity.ORDER_PARAMETER: PlotDefault(cmap="viridis", clabel="$|\\psi|$"),
    Quantity.PHASE: PlotDefault(
        cmap="twilight_shifted",
        clabel="$\\arg(\\psi)/\\pi$",
    ),
    Quantity.SUPERCURRENT: PlotDefault(cmap="inferno", clabel="$|\\vec{{J}}_s|/J_0$"),
    Quantity.NORMAL_CURRENT: PlotDefault(cmap="inferno", clabel="$|\\vec{{J}}_n|/J_0$"),
    Quantity.SCALAR_POTENTIAL: PlotDefault(cmap="magma", clabel="$\\mu/v_0$"),
    Quantity.APPLIED_VECTOR_POTENTIAL: PlotDefault(
        cmap="cividis", clabel="$a_\\mathrm{{applied}}/(\\xi B_{{c2}})$"
    ),
    Quantity.INDUCED_VECTOR_POTENTIAL: PlotDefault(
        cmap="cividis", clabel="$a_\\mathrm{{induced}}/(\\xi B_{{c2}})$"
    ),
    Quantity.ALPHA: PlotDefault(cmap="viridis", clabel="$\\alpha$"),
    Quantity.VORTICITY: PlotDefault(
        cmap="coolwarm", clabel="$(\\vec{{\\nabla}}\\times\\vec{{J}})\\cdot\\hat{{z}}$"
    ),
}
