from dataclasses import dataclass
from enum import Enum
from typing import Sequence


class Observable(Enum):
    COMPLEX_FIELD = "Complex field"
    PHASE = "Phase"
    SUPERCURRENT = "Supercurrent density"
    NORMAL_CURRENT = "Normal current density"
    VORTICITY = "Vorticity"
    SCALAR_POTENTIAL = "Scalar potential"
    APPLIED_VECTOR_POTENTIAL = "Applied vector potential"
    INDUCED_VECTOR_POTENTIAL = "Induced vector potential"
    TOTAL_VECTOR_POTENTIAL = "Total vector potential"
    ALPHA = "Alpha"

    @classmethod
    def get_keys(cls) -> Sequence[str]:
        return list(item.name for item in Observable)

    @classmethod
    def from_key(cls, key: str) -> "Observable":
        return Observable[key.upper()]


colormaps = {
    Observable.COMPLEX_FIELD: "viridis",
    Observable.PHASE: "twilight_shifted",
    Observable.SUPERCURRENT: "inferno",
    Observable.NORMAL_CURRENT: "inferno",
    Observable.SCALAR_POTENTIAL: "magma",
    Observable.TOTAL_VECTOR_POTENTIAL: "cividis",
    Observable.APPLIED_VECTOR_POTENTIAL: "cividis",
    Observable.INDUCED_VECTOR_POTENTIAL: "cividis",
    Observable.ALPHA: "viridis",
    Observable.VORTICITY: "coolwarm",
}


@dataclass
class PlotDefault:
    cmap: str
    clabel: str
    xlabel: str = "$x/\\xi$"
    ylabel: str = "$y/\\xi$"


PLOT_DEFAULTS = {
    Observable.COMPLEX_FIELD: PlotDefault(cmap="viridis", clabel="$|\\psi|$"),
    Observable.PHASE: PlotDefault(
        cmap="twilight_shifted",
        clabel="$\\arg(\\psi)/\\pi$",
    ),
    Observable.SUPERCURRENT: PlotDefault(cmap="inferno", clabel="$|\\vec{{J}}_s|/J_0$"),
    Observable.NORMAL_CURRENT: PlotDefault(
        cmap="inferno", clabel="$|\\vec{{J}}_n|/J_0$"
    ),
    Observable.SCALAR_POTENTIAL: PlotDefault(cmap="magma", clabel="$\\mu/v_0$"),
    Observable.TOTAL_VECTOR_POTENTIAL: PlotDefault(
        cmap="cividis", clabel="$a/(\\xi B_{{c2}})$"
    ),
    Observable.APPLIED_VECTOR_POTENTIAL: PlotDefault(
        cmap="cividis", clabel="$a_\\mathrm{{applied}}/(\\xi B_{{c2}})$"
    ),
    Observable.INDUCED_VECTOR_POTENTIAL: PlotDefault(
        cmap="cividis", clabel="$a_\\mathrm{{induced}}/(\\xi B_{{c2}})$"
    ),
    Observable.ALPHA: PlotDefault(cmap="viridis", clabel="$\\alpha$"),
    Observable.VORTICITY: PlotDefault(
        cmap="coolwarm", clabel="$(\\vec{{\\nabla}}\\times\\vec{{J}})\\cdot\\hat{{z}}$"
    ),
}
