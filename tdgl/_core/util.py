from typing import Tuple

import numpy as np


def sum_contributions(
    groups: np.ndarray, values: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sum all contributions from value over each group.

    Args:
        groups: The groups to get the summed values for. Must be a one
            dimensional vector.
        values: The values for each item. Must be a one dimensional vector.

    Returns:
        A tuple of groups, group values, and counts.
    """

    # Get the unique groups and the corresponding indices
    unique_groups, idx, counts = np.unique(
        groups, return_inverse=True, return_counts=True
    )

    # Sum each unique group
    group_values = np.bincount(idx, weights=values)

    return unique_groups, group_values, counts
