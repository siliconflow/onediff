"""Hijack utils for stable-diffusion."""
import importlib
import inspect
from types import FunctionType
from typing import Callable, List, Union
from collections import deque

__all__ = ["Hijacker", "hijack_func"]


def get_func_full_name(func: FunctionType):
    """Get the full name of a function."""
    module = inspect.getmodule(func)
    if module is None:
        raise ValueError(f"Cannot get module of function {func}")
    return f"{module.__name__}.{func.__qualname__}"


class CondFuncWrapper:
    def __init__(self, cond_func_instance):
        self.cond_func_instance = cond_func_instance

    def __call__(self, *args, **kwargs):
        return self.cond_func_instance(*args, **kwargs)

    def add_condition(self, sub_func: FunctionType, cond_func: FunctionType, last=True):
        """Pairs are returned in LIFO order if last is true or FIFO order if false."""
        instance: CondFunc = self.cond_func_instance
        if last:
            instance._sub_funcs.append(sub_func)
            instance._cond_funcs.append(cond_func)
        else:
            instance._sub_funcs.appendleft(sub_func)
            instance._cond_funcs.appendleft(cond_func)


class CondFunc:
    """A function that conditionally calls another function.

    Copied from: https://github.com/AUTOMATIC1111/stable-diffusion-webui/blob/master/modules/sd_hijack_utils.py
    """

    def __new__(
        cls,
        orig_func: Union[str, Callable],
        sub_funcs: List[FunctionType],
        cond_funcs: List[FunctionType],
    ):
        # self: CondFunc instance
        self = super(CondFunc, cls).__new__(cls)
        if isinstance(orig_func, FunctionType):
            orig_func = get_func_full_name(orig_func)
        assert isinstance(orig_func, str)

        func_path = orig_func.split(".")
        for i in range(len(func_path) - 1, -1, -1):
            try:
                resolved_obj = importlib.import_module(".".join(func_path[:i]))
                break
            except ImportError:
                pass

        if resolved_obj is None:
            raise ImportError(f"Could not resolve module for {func_path}")

        for attr_name in func_path[i:-1]:
            resolved_obj = getattr(resolved_obj, attr_name)
        orig_func = getattr(resolved_obj, func_path[-1])
        wrapper = CondFuncWrapper(self)
        setattr(
            resolved_obj, func_path[-1], wrapper,
        )

        def unhijack_func():
            setattr(resolved_obj, func_path[-1], orig_func)

        self.__init__(orig_func, sub_funcs, cond_funcs)

        return (wrapper, unhijack_func)

    def __init__(
        self,
        orig_func: Callable,
        sub_funcs: List[FunctionType],
        cond_funcs: List[FunctionType],
    ):
        self._orig_func = orig_func
        self._sub_funcs = deque(sub_funcs)
        self._cond_funcs = deque(cond_funcs)

    def __call__(self, *args, **kwargs):
        for cond_func, sub_func in zip(self._cond_funcs, self._sub_funcs):
            if not cond_func or cond_func(self._orig_func, *args, **kwargs):
                return sub_func(self._orig_func, *args, **kwargs)
        else:
            return self._orig_func(*args, **kwargs)


def ensure_list(obj: Union[FunctionType, List[FunctionType]]) -> List[FunctionType]:
    if not isinstance(obj, list):
        return [obj]
    return obj


def hijack_func(
    orig_func: Union[str, Callable],
    sub_func: Callable,
    cond_func: Callable,
    *,
    last=True,
):
    """
    Hijacks a function with another function.

    Returns:
        A tuple of (hijacked_func, unhijack_func)

    Examples:
        >>> def foo(*args, **kwargs):
        >>>     # orig_func
        >>>     print('foo')
        >>> def bar(orig_func, *args, **kwargs):
        >>>     # sub_func
        >>>     print('bar')
        >>> def cond_func(orig_func, *args, **kwargs):
        >>>     # cond_func
        >>>     return True
        >>> hijack_func(foo, bar, cond_func)
        >>> foo()
        bar
    """
    if isinstance(orig_func, CondFuncWrapper):
        orig_func.add_condition(sub_func, cond_func, last=last)
        return orig_func

    if isinstance(orig_func, FunctionType):
        orig_func = get_func_full_name(orig_func)
    return CondFunc(orig_func, ensure_list(sub_func), ensure_list(cond_func))


class Hijacker:
    """A class that hijacks a list of functions.

    Example:
        >>> hijacker = Hijacker([
        >>>     ("module.orig_func1", sub_func1, cond_func1),
        >>>     ("module.orig_func2", sub_func2, cond_func2),
        >>>     # Add more tuples as needed
        >>> ])
    """

    def __init__(self, funcs_list=[]):
        self.funcs_list = funcs_list
        self.unhijack_funcs = []

    def hijack(self, last=True):
        self.unhijack()
        for orig_func, sub_func, cond_func in self.funcs_list:
            _, unhijack_func = hijack_func(orig_func, sub_func, cond_func, last=last)
            self.unhijack_funcs.append(unhijack_func)

    def unhijack(self):
        if len(self.unhijack_funcs) > 0:
            for unhijack_func in self.unhijack_funcs:
                unhijack_func()
            self.unhijack_funcs = []

    def extend_unhijack(self, unhijack_func):
        self.unhijack_funcs.append(unhijack_func)

    def register(
        self, orig_func: FunctionType, sub_func: Callable, cond_func: Callable
    ):
        self.funcs_list.append((orig_func, sub_func, cond_func))


if __name__ == "__main__":

    def orig_func(*args, **kwargs):
        print("Original function")

    def sub_func_0(orig_func, *args, **kwargs):
        print(f"Called sub_func_0")

    def sub_func_1(orig_func, *args, **kwargs):
        print(f"Called sub_func_1")

    cond_0 = True

    def cond_func_0(orig_func, *args, **kwargs):
        return cond_0

    def cond_func_1(orig_func, *args, **kwargs):
        return True

    hijack_func(orig_func, sub_func_0, cond_func_0)
    orig_func()  # Output: Called sub_func_0

    hijack_func(orig_func, sub_func_1, cond_func_1)
    cond_0 = False
    orig_func()  # Output: Called sub_func_1
    cond_0 = True
    orig_func()  # Called sub_func_0
    hijack_func(orig_func, sub_func_1, cond_func_1, last=False)
    cond_0 = True
    orig_func()  # Called sub_func_1
