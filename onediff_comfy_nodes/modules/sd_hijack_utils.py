"""Hijack utils for stable-diffusion."""
import importlib
import inspect
from collections import deque
from types import FunctionType
from typing import Callable, List, Union

__all__ = ["Hijacker", "hijack_func"]


def get_func_full_name(func: FunctionType):
    """Get the full name of a function."""
    module = inspect.getmodule(func)
    if module is None:
        raise ValueError(f"Cannot get module of function {func}")
    return f"{module.__name__}.{func.__qualname__}"


class CondFunc:
    """A function that conditionally calls another function.

    Copied from: https://github.com/AUTOMATIC1111/stable-diffusion-webui/blob/master/modules/sd_hijack_utils.py
    """

    # Dictionary to store hijacked methods and their corresponding CondFunc instances
    hijacked_registry = {}

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

        def hijacked_method(*args, **kwargs):
            return self(*args, **kwargs)

        setattr(
            resolved_obj,
            func_path[-1],
            hijacked_method,
        )

        def unhijack_func():
            setattr(resolved_obj, func_path[-1], orig_func)
            del cls.hijacked_registry[hijacked_method]

        self.__init__(orig_func, sub_funcs, cond_funcs)
        cls.hijacked_registry[hijacked_method] = self
        return (hijacked_method, unhijack_func)

    @staticmethod
    def is_hijacked_method(func: Callable):
        return func in CondFunc.hijacked_registry

    @staticmethod
    def get_hijacked_instance(func: Callable):
        return CondFunc.hijacked_registry.get(func)

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

    def add_condition(self, sub_func: FunctionType, cond_func: FunctionType, last=True):
        """Pairs are returned in LIFO order if last is true or FIFO order if false."""
        instance: CondFunc = self
        if last:
            instance._sub_funcs.append(sub_func)
            instance._cond_funcs.append(cond_func)
        else:
            instance._sub_funcs.appendleft(sub_func)
            instance._cond_funcs.appendleft(cond_func)


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
    if CondFunc.is_hijacked_method(orig_func):
        ins = CondFunc.get_hijacked_instance(orig_func)
        ins.add_condition(sub_func, cond_func, last=last)
        return orig_func, lambda: None

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
        if funcs_list and isinstance(funcs_list, List):
            self.funcs_list = funcs_list.copy()
        else:
            self.funcs_list = []
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
        return "orig_func"

    def sub_func_0(orig_func, *args, **kwargs):
        print(f"Called sub_func_0")
        return "sub_func_0"

    def sub_func_1(orig_func, *args, **kwargs):
        print(f"Called sub_func_1")
        return "sub_func_1"

    cond_0 = True

    def cond_func_0(orig_func, *args, **kwargs):
        return cond_0

    def cond_func_1(orig_func, *args, **kwargs):
        return True

    hijack_func(orig_func, sub_func_0, cond_func_0)
    assert orig_func() == "sub_func_0"  # Output: Called sub_func_0

    hijack_func(orig_func, sub_func_1, cond_func_1)
    cond_0 = False
    assert orig_func() == "sub_func_1"  # Output: Called sub_func_1
    cond_0 = True
    assert orig_func() == "sub_func_0"  # Called sub_func_0
    hijack_func(orig_func, sub_func_1, cond_func_1, last=False)
    cond_0 = True
    assert orig_func() == "sub_func_1"  # Called sub_func_1

    class Case1:
        def clone(self):
            print(f"{type(self)}.clone")

    def cond_func(org_fn, self):
        return True

    def custom_clone(org_fn, self):
        print(f"custom_clone")
        return "custom_clone"

    hijack_func(Case1.clone, custom_clone, cond_func)

    def custom_clone_1(org_fn, self):
        print(f"custom_clone_1")
        return "custom_clone_1"

    assert Case1().clone() == "custom_clone"
    hijack_func(Case1.clone, custom_clone_1, cond_func, last=False)

    assert Case1().clone() == "custom_clone_1"
