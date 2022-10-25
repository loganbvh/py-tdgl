from typing import Dict, Sequence, Union

import numpy as np


class RunningState:
    """
    Storage class for saving data that should be saved each time step. Used
    for IV curves or simular.

    Args:
        names: Names of the parameters to be saved.
        buffer: Size of the buffer.
    """

    def __init__(self, names: Sequence[str], buffer: int):
        self.step = 0
        self.buffer = buffer
        self.values = dict((name, np.zeros(buffer)) for name in names)

    def next(self) -> None:
        """Go to the next step."""
        self.step += 1

    def set_step(self, step: int) -> None:
        """Set the current step.

        Args:
            step: Step to go to.
        """
        self.step = step

    def clear(self) -> None:
        """Clear the buffer."""

        self.step = 0
        for key in self.values:
            self.values[key] = np.zeros(self.buffer)

    def append(self, name: str, value: Union[float, int]) -> None:
        """Append data to the buffer.

        Args:
            name: Data to append.
            value: Value of the data.
        """

        self.values[name][self.step] = value

    def export(self) -> Dict[str, np.ndarray]:
        """Export data to save to disk.

        Returns:
            A dict with the data.
        """

        return self.values
