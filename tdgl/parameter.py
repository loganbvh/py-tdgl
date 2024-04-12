import hashlib
import inspect
import operator
from numbers import Number
from typing import Callable, Optional, Union

import cloudpickle
import numpy as np


class _FakeArgSpec:
    def __init__(
        self,
        args=None,
        varargs=None,
        varkw=None,
        defaults=None,
        kwonlyargs=None,
        kwonlydefaults=None,
        annotations=None,
    ):
        self.args = args
        self.varargs = varargs
        self.varkw = varkw
        self.defaults = defaults
        self.kwonlyargs = kwonlyargs
        self.kwonlydefaults = kwonlydefaults
        self.annotations = annotations


def function_repr(
    func: Callable,
    argspec: Optional[Union[_FakeArgSpec, inspect.FullArgSpec]] = None,
) -> str:
    """Returns a human-readable string representation for a function."""
    if argspec is None:
        argspec = inspect.getfullargspec(func)
    args = [str(arg) for arg in argspec.args]

    if argspec.defaults:
        for i, val in enumerate(argspec.defaults[::-1]):
            args[-(i + 1)] = args[-(i + 1)] + f"={val!r}"

    if argspec.varargs:
        args.append("*" + argspec.varargs)

    if argspec.kwonlyargs:
        if not argspec.varargs:
            args.append("*")
        args.extend(argspec.kwonlyargs)
    if argspec.kwonlydefaults:
        for i, name in enumerate(args):
            if name in argspec.kwonlydefaults:
                args[i] = args[i] + f"={argspec.kwonlydefaults[name]!r}"
    if argspec.varkw:
        args.append("**" + argspec.varkw)

    if argspec.annotations:
        for i, name in enumerate(args):
            if name in argspec.annotations:
                args[i] = args[i] + f": {argspec.annotations[name].__name__!r}"

    return func.__name__ + "(" + ", ".join(args) + ")"


class Parameter:
    """A callable object that computes a scalar or vector quantity
    as a function of position coordinates x, y (and optionally z and time t).

    Addition, subtraction, multiplication, and division
    between multiple Parameters and/or numbers is supported.
    The result of any of these operations is a ``CompositeParameter`` object.

    Args:
        func: A callable/function that actually calculates the parameter's value.
            The function must take x, y (and optionally z) as the first and only
            positional arguments, and all other arguments must be keyword arguments.
            Therefore func should have a signature like
            ``func(x, y, z, a=1, b=2, c=True)``, ``func(x, y, *, a, b, c)``,
            ``func(x, y, z, *, a, b, c)``, or ``func(x, y, z, *, a, b=None, c=3)``.
            For time-dependent Parameters, ``func`` must also take time ``t`` as a
            keyword-only argument.
        time_dependent: Specifies that ``func`` is a function of time ``t``.
        kwargs: Keyword arguments for func.
    """

    __slots__ = ("func", "kwargs", "time_dependent", "_cache", "_use_cache")

    def __init__(self, func: Callable, time_dependent: bool = False, **kwargs):
        self._use_cache = kwargs.pop("use_cache", None)
        argspec = inspect.getfullargspec(func)
        args = argspec.args
        num_args = 2
        if args[:num_args] != ["x", "y"]:
            raise ValueError(
                "The first function arguments must be x and y, "
                f"not {', '.join(args[:num_args])!r}."
            )
        if "z" in args:
            if args.index("z") != num_args:
                raise ValueError(
                    "If the function takes an argument z, "
                    "it must be the third argument (x, y, z)."
                )
            num_args = 3
        defaults = argspec.defaults or []
        if len(defaults) != len(args) - num_args:
            raise ValueError(
                "All arguments other than x, y, z must be keyword arguments."
            )
        self.time_dependent = time_dependent
        defaults_dict = dict(zip(args[num_args:], defaults))
        kwonlyargs = set(kwargs) - set(argspec.args[num_args:])
        if not kwonlyargs.issubset(set(argspec.kwonlyargs or [])):
            raise ValueError(
                f"Provided keyword-only arguments ({kwonlyargs!r}) "
                f"do not match the function signature: {function_repr(func)}."
            )
        defaults_dict.update(argspec.kwonlydefaults or {})

        self.func = func
        self.kwargs = defaults_dict
        self.kwargs.update(kwargs)
        self._cache = {}

        if self.time_dependent and "t" not in argspec.kwonlyargs:
            raise ValueError(
                "A time-dependent Parameter must take time t as a keyword argument."
            )

    def _hash_args(self, x, y, z, t) -> str:
        def _coerce_to_tuple(a):
            try:
                return tuple(_coerce_to_tuple(i) for i in a)
            except TypeError:
                return a

        def _to_tuple(items):
            results = []
            for key, value in items:
                if isinstance(value, dict):
                    value = _to_tuple(value.items())
                elif isinstance(value, (list, np.ndarray)):
                    value = _coerce_to_tuple(value)
                results.append((key, value))
            return tuple(results)

        return (
            hex(hash(_to_tuple(self.kwargs.items())))
            + hashlib.sha1(np.ascontiguousarray(x)).hexdigest()
            + hashlib.sha1(np.ascontiguousarray(y)).hexdigest()
            + hashlib.sha1(np.ascontiguousarray(z)).hexdigest()
            + hex(hash(t))
        )

    def _evaluate(
        self,
        x: Union[Number, np.ndarray],
        y: Union[Number, np.ndarray],
        z: Optional[Union[Number, np.ndarray]] = None,
        t: Optional[float] = None,
    ) -> Union[Number, np.ndarray]:
        kwargs = self.kwargs.copy()
        if t is not None:
            kwargs["t"] = t
        x, y = np.atleast_1d(x, y)
        if z is not None:
            kwargs["z"] = np.atleast_1d(z)
        result = np.asarray(self.func(x, y, **kwargs)).squeeze()
        if result.ndim == 0:
            result = result.item()
        return result

    def __call__(
        self,
        x: Union[Number, np.ndarray],
        y: Union[Number, np.ndarray],
        z: Optional[Union[Number, np.ndarray]] = None,
        t: Optional[float] = None,
    ) -> Union[Number, np.ndarray]:
        if self._use_cache:
            cache_key = self._hash_args(x, y, z, t)
            if cache_key not in self._cache:
                self._cache[cache_key] = self._evaluate(x, y, z, t)
            return self._cache[cache_key]
        return self._evaluate(x, y, z, t)

    def _clear_cache(self) -> None:
        self._cache.clear()

    def _get_argspec(self) -> _FakeArgSpec:
        if self.kwargs:
            kwargs, kwarg_values = list(zip(*self.kwargs.items()))
        else:
            kwargs = []
            kwarg_values = []
        kwargs = list(kwargs)
        kwarg_values = list(kwarg_values)
        if self.time_dependent:
            kwargs.insert(0, "time_dependent")
            kwarg_values.insert(0, True)
        return _FakeArgSpec(args=kwargs, defaults=kwarg_values)

    def __repr__(self) -> str:
        func_repr = function_repr(self.func, argspec=self._get_argspec())
        return f"{self.__class__.__name__}<{func_repr}>"

    def __add__(self, other) -> "CompositeParameter":
        """self + other"""
        return CompositeParameter(self, other, operator.add)

    def __radd__(self, other) -> "CompositeParameter":
        """other + self"""
        return CompositeParameter(other, self, operator.add)

    def __sub__(self, other) -> "CompositeParameter":
        """self - other"""
        return CompositeParameter(self, other, operator.sub)

    def __rsub__(self, other) -> "CompositeParameter":
        """other - self"""
        return CompositeParameter(other, self, operator.sub)

    def __mul__(self, other) -> "CompositeParameter":
        """self * other"""
        return CompositeParameter(self, other, operator.mul)

    def __rmul__(self, other) -> "CompositeParameter":
        """other * self"""
        return CompositeParameter(other, self, operator.mul)

    def __truediv__(self, other) -> "CompositeParameter":
        """self / other"""
        return CompositeParameter(self, other, operator.truediv)

    def __rtruediv__(self, other) -> "CompositeParameter":
        """other / self"""
        return CompositeParameter(other, self, operator.truediv)

    def __pow__(self, other) -> "CompositeParameter":
        """self ** other"""
        return CompositeParameter(self, other, operator.pow)

    def __rpow__(self, other) -> "CompositeParameter":
        """other ** self"""
        return CompositeParameter(other, self, operator.pow)

    def __eq__(self, other) -> bool:
        if other is self:
            return True

        if not isinstance(other, Parameter):
            return False

        # Check if function bytecode is the same
        if self.func.__code__ != other.func.__code__:
            return False

        if set(self.kwargs) != set(other.kwargs):
            return False

        def array_safe_equals(a, b) -> bool:
            """Check if a and b are equal, even if they are numpy arrays."""
            if a is b:
                return True
            if isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
                return a.shape == b.shape and np.allclose(a, b)
            try:
                return a == b
            except TypeError:
                return NotImplemented

        for key in self.kwargs:
            if not array_safe_equals(self.kwargs[key], other.kwargs[key]):
                return False

        return True


class CompositeParameter(Parameter):
    """A callable object that behaves like a Parameter
    (i.e. it computes a scalar or vector quantity as a function of
    position coordinates x, y, z). A CompositeParameter object is created as
    a result of mathematical operations between Parameters, CompositeParameters,
    and/or numbers.

    Addition, subtraction, multiplication, division, and exponentiation
    between ``Parameters``, ``CompositeParameters`` and numbers are supported.
    The result of any of these operations is a new ``CompositeParameter`` object.

    Args:
        left: The object on the left-hand side of the operator.
        right: The object on the right-hand side of the operator.
        operator_: The operator acting on left and right (or its string representation).
    """

    VALID_OPERATORS = {
        operator.add: "+",
        operator.sub: "-",
        operator.mul: "*",
        operator.truediv: "/",
        operator.pow: "**",
    }

    def __init__(
        self,
        left: Union[Number, Parameter, "CompositeParameter"],
        right: Union[Number, Parameter, "CompositeParameter"],
        operator_: Union[Callable, str],
    ):
        valid_types = (Number, Parameter, CompositeParameter)
        if not isinstance(left, valid_types):
            raise TypeError(
                f"Left must be a number, Parameter, or CompositeParameter, "
                f"not {type(left)!r}."
            )
        if not isinstance(right, valid_types):
            raise TypeError(
                f"Right must be a number, Parameter, or CompositeParameter, "
                f"not {type(right)!r}."
            )
        if isinstance(left, Number) and isinstance(right, Number):
            raise TypeError(
                "Either left or right must be a Parameter or CompositeParameter."
            )
        if isinstance(operator_, str):
            operators = {v: k for k, v in self.VALID_OPERATORS.items()}
            operator_ = operators.get(operator_.strip(), None)
        if operator_ not in self.VALID_OPERATORS:
            raise ValueError(
                f"Unknown operator, {operator_!r}. "
                f"Valid operators are {list(self.VALID_OPERATORS)!r}."
            )
        self._cache = {}
        self.left = left
        self.right = right
        self.operator = operator_
        self.time_dependent = False
        if isinstance(self.left, Parameter) and self.left.time_dependent:
            self.time_dependent = True
            if self.left._use_cache is None:
                self.left._use_cache = True
        if isinstance(self.right, Parameter) and self.right.time_dependent:
            self.time_dependent = True
            if self.right._use_cache is None:
                self.right._use_cache = True

    def _clear_cache(self) -> None:
        self._cache.clear()
        if isinstance(self.right._cache, Parameter):
            self.right._clear_cache()
        if isinstance(self.left, Parameter):
            self.left._clear_cache()

    def __call__(
        self,
        x: Union[Number, np.ndarray],
        y: Union[Number, np.ndarray],
        z: Union[Number, np.ndarray, None] = None,
        t: Optional[float] = None,
    ) -> Union[Number, np.ndarray]:
        kwargs = dict() if t is None else dict(t=t)
        values = []
        for operand in (self.left, self.right):
            if isinstance(operand, Parameter):
                if operand.time_dependent:
                    value = operand(x, y, z, **kwargs)
                else:
                    value = operand(x, y, z)
            else:
                value = operand
            values.append(value)
        return self.operator(*values)

    def _bare_repr(self) -> str:
        op_str = self.VALID_OPERATORS[self.operator]
        if isinstance(self.left, CompositeParameter):
            left_repr = self.left._bare_repr()
        elif isinstance(self.left, Parameter):
            left_argspec = self.left._get_argspec()
            left_repr = function_repr(self.left.func, left_argspec)
        else:
            left_repr = str(self.left)

        if isinstance(self.right, CompositeParameter):
            right_repr = self.right._bare_repr()
        elif isinstance(self.right, Parameter):
            right_argspec = self.right._get_argspec()
            right_repr = function_repr(self.right.func, right_argspec)
        else:
            right_repr = str(self.right)

        return f"({left_repr} {op_str} {right_repr})"

    def __eq__(self, other) -> bool:
        if other is self:
            return True

        if not isinstance(other, type(self)):
            return False

        return (
            self.left == other.left
            and self.right == other.right
            and self.operator is other.operator
        )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}<{self._bare_repr()}>"

    def __getstate__(self):
        state = self.__dict__.copy()
        state["left"] = cloudpickle.dumps(state["left"])
        state["right"] = cloudpickle.dumps(state["right"])
        return state

    def __setstate__(self, state):
        state["left"] = cloudpickle.loads(state["left"])
        state["right"] = cloudpickle.loads(state["right"])
        self.__dict__.update(state)


class Constant(Parameter):
    """A Parameter whose value doesn't depend on position or time."""

    def __init__(self, value: Number, dimensions: int = 2):
        if dimensions not in (2, 3):
            raise ValueError(f"Dimensions must be 2 or 3, got {dimensions}.")
        if dimensions == 2:

            def constant(x, y, value=0):
                return value * np.ones_like(x)

        else:

            def constant(x, y, z, value=0):
                return value * np.ones_like(x)

        super().__init__(constant, value=value)
