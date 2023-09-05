from ..parameter import Parameter


def linear_ramp(x, y, z, *, t, tmin, tmax, initial: float = 0.0, final: float = 1.0):
    """A linear ramp from ``initial`` to ``final`` over time period ``tmin`` to ``tmax``.

    For times less than ``tmin``, the value stays at ``initial``. For times greater
    than ``tmax``, the value stays at ``final``.
    """
    if t < tmin:
        return initial
    elif t < tmax:
        return initial + (final - initial) * (t - tmin) / (tmax - tmin)
    return final


def LinearRamp(*, tmin: float, tmax, initial: float = 0.0, final: float = 1.0):
    """A linear ramp from ``initial`` to ``final`` over time period ``tmin`` to ``tmax``.

    For times less than ``tmin``, the value stays at ``initial``. For times greater
    than ``tmax``, the value stays at ``final``.

    Args:
        tmin: Ramp start time
        tmax: Ramp end time
        initial: Ramp initial value
        final: Ramp final value

    Returns:
        A :class:`tdgl.Parameter` that produces a linear ramp.
    """
    return Parameter(
        linear_ramp,
        tmin=tmin,
        tmax=tmax,
        initial=initial,
        final=final,
        time_dependent=True,
    )


def Scale(func, **kwargs):
    """An arbitrary time-dependent scale factor.

    Args:
        func: A callable with signature func(x, y, z, *, t, **kwargs) that
            produces a time-dependent scale factor.

    Returns:
        A :class:`tdgl.Parameter` that evaluates ``func``
    """
    kwargs["time_dependent"] = True
    return Parameter(func, **kwargs)
