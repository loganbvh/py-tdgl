from typing import Union

import h5py


class Layer:
    """A superconducting thin film.

    Args:
        london_lambda: The London penetration depth of the film.
        coherence_length: The superconducting coherence length of the film.
        thickness: The thickness of the film.
        conductivity: The normal state conductivity of the superconductor in
            Siemens / length_unit.
        u: The ratio of the relaxation times for the order parameter amplitude
            and phase. This value is 5.79 for dirty superconductors.
        gamma: This parameter quantifies the effect of inelastic phonon-electron
            scattering. :math:`\\gamma` is proportional to the inelastic scattering
            time and the size of the superconducting gap.
        z0: Vertical location of the film.
    """

    def __init__(
        self,
        *,
        london_lambda: float,
        coherence_length: float,
        thickness: float,
        conductivity: Union[float, None] = None,
        u: float = 5.79,
        gamma: float = 10.0,
        z0: float = 0,
    ):
        self.london_lambda = london_lambda
        self.coherence_length = coherence_length
        self.thickness = thickness
        self.conductivity = conductivity
        self.u = u
        self.gamma = gamma
        self.z0 = z0

    @property
    def Lambda(self) -> float:
        """Effective magnetic penetration depth, :math:`\\Lambda=\\lambda^2/d`."""
        return self.london_lambda**2 / self.thickness

    def copy(self) -> "Layer":
        """Create a deep copy of the :class:`tdgl.Layer`."""
        return Layer(
            london_lambda=self.london_lambda,
            coherence_length=self.coherence_length,
            thickness=self.thickness,
            conductivity=self.conductivity,
            u=self.u,
            gamma=self.gamma,
            z0=self.z0,
        )

    def to_hdf5(self, h5_group: h5py.Group) -> None:
        """Save the :class:`tdgl.Layer` to an :class:`h5py.Group`.

        Args:
            h5_group: An open :class:`h5py.Group` to which to save the layer.
        """
        h5_group.attrs["london_lambda"] = self.london_lambda
        h5_group.attrs["coherence_length"] = self.coherence_length
        h5_group.attrs["thickness"] = self.thickness
        h5_group.attrs["u"] = self.u
        h5_group.attrs["gamma"] = self.gamma
        h5_group.attrs["z0"] = self.z0
        if self.conductivity is not None:
            h5_group.attrs["conductivity"] = self.conductivity

    @staticmethod
    def from_hdf5(h5_group: h5py.Group) -> "Layer":
        """Load a :class:`tdgl.Layer` from an :class:`h5py.Group`.

        Args:
            h5_group: An open :class:`h5py.Group` from which to load the layer.

        Returns:
            A new :class:`tdgl.Layer` instance.
        """

        def get(key, default=None):
            if key in h5_group.attrs:
                return h5_group.attrs[key]
            return default

        return Layer(
            london_lambda=get("london_lambda"),
            coherence_length=get("coherence_length"),
            thickness=get("thickness"),
            conductivity=get("conductivity"),
            u=get("u"),
            gamma=get("gamma"),
            z0=get("z0"),
        )

    def __eq__(self, other):
        if self is other:
            return True

        if not isinstance(other, Layer):
            return False

        return (
            self.london_lambda == other.london_lambda
            and self.coherence_length == other.coherence_length
            and self.thickness == other.thickness
            and self.conductivity == other.conductivity
            and self.u == other.u
            and self.gamma == other.gamma
            and self.z0 == other.z0
        )

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"london_lambda={self.london_lambda}, "
            f"coherence_length={self.coherence_length}, "
            f"thickness={self.thickness}, "
            f"conducivitiy={self.conductivity}, "
            f"u={self.u}, "
            f"gamma={self.gamma}, "
            f"z0={self.z0}"
            f")"
        )
