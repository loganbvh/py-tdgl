from typing import List

from superscreen import Layer


def ibm_squid_layers(
    align: str = "middle",
    london_lambda: float = 0.08,
    z0: float = 0.0,
    d_BE: float = 0.16,
    d_I1: float = 0.15,
    d_W1: float = 0.10,
    d_I2: float = 0.13,
    d_W2: float = 0.20,
) -> List[Layer]:
    """Return a list of superscreen.Layers representing the superconducting layers
    in IBM SQUID susceptometers.

    See https://arxiv.org/pdf/1605.09483.pdf, Figure 8.

    Args:
        align: Whether to position the 2D model layer at the top, middle, or bottom
            of the phyical 3D metal layer.
        london_lambda: The London penetration depth for the superconducting films,
            in microns.
        z0: The vertical position of the bottom of W2, i.e. the surface of the
            SQUID chip.
        d_BE, d_I1, d_W1, d_I2, d_W2: Layer thicknesses in microns.

    Returns:
        A list a Layer objects representing the SQUID wiring layers.

    """
    assert align in ("top", "middle", "bottom")

    # Metal layer vertical positions in microns.
    scale = {"bottom": 0, "middle": 0.5, "top": 1}[align]
    z0_W2 = z0 + scale * d_W2
    z0_W1 = z0 + d_W2 + d_I2 + scale * d_W1
    z0_BE = z0 + d_W2 + d_I2 + d_W1 + d_I1 + scale * d_W2

    return [
        Layer("W2", london_lambda=london_lambda, thickness=d_W2, z0=z0_W2),
        Layer("W1", london_lambda=london_lambda, thickness=d_W1, z0=z0_W1),
        Layer("BE", london_lambda=london_lambda, thickness=d_BE, z0=z0_BE),
    ]
