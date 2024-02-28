__all__ = ["basic_type_proxy"]


def basic_type_proxy(value):
    if not isinstance(value, (int, float, str, bool)):
        raise ValueError(
            f"Value must be of type int, float, str or bool. Got {type(value)}"
        )

    # TODO support for bool type
    if isinstance(value, bool):
        # TypeError: type 'bool' is not an acceptable base type
        return value

    class BasicTypeProxy(type(value)):
        """A proxy class for basic types (int, float, str.) 
        
        This class is used to wrap basic types and allow for easy modification by copy_ method.
        """

        def __init__(self, value):
            self._value = value

        @property
        def __class__(self):
            return type(self._value)

        def copy_(self, value):
            self._value = value

        def __getattr__(self, name):
            if name in ["_value", "copy_"]:
                return super().__getattribute__(name)
            return getattr(self._value, name)

        def __setattr__(self, name, value):
            if name == "_value":
                super().__setattr__(name, value)
            else:
                setattr(self._value, name, value)

        def __add__(self, other):
            if isinstance(other, BasicTypeProxy):
                return self._value + other._value
            return self._value + other

        def __radd__(self, other):
            if isinstance(other, BasicTypeProxy):
                return other._value + self._value
            return other + self._value

        def __sub__(self, other):
            if isinstance(other, BasicTypeProxy):
                return self._value - other._value
            return self._value - other

        def __rsub__(self, other):
            if isinstance(other, BasicTypeProxy):
                return other._value - self._value
            return other - self._value

        def __mul__(self, other):
            if isinstance(other, BasicTypeProxy):
                return self._value * other._value
            return self._value * other

        def __rmul__(self, other):
            if isinstance(other, BasicTypeProxy):
                return other._value * self._value
            return other * self._value

        def __truediv__(self, other):
            if isinstance(other, BasicTypeProxy):
                return self._value / other._value
            return self._value / other

        def __rtruediv__(self, other):
            if isinstance(other, BasicTypeProxy):
                return other._value / self._value
            return other / self._value

        def __bool__(self):
            return self._value

        def __pow__(self, other):
            if isinstance(other, BasicTypeProxy):
                return self._value ** other._value
            return self._value ** other

        def __rpow__(self, other):
            if isinstance(other, BasicTypeProxy):
                return other._value ** self._value
            return other ** self._value

        def __floordiv__(self, other):
            if isinstance(other, BasicTypeProxy):
                return self._value // other._value
            return self._value // other

        def __rfloordiv__(self, other):
            if isinstance(other, BasicTypeProxy):
                return other._value // self._value
            return other // self._value

        def __mod__(self, other):
            if isinstance(other, BasicTypeProxy):
                return self._value % other._value
            return self._value % other

        def __rmod__(self, other):
            if isinstance(other, BasicTypeProxy):
                return other._value % self._value
            return other % self._value

        def __eq__(self, other):
            if isinstance(other, BasicTypeProxy):
                return self._value == other._value
            return self._value == other

        def __ne__(self, other):
            if isinstance(other, BasicTypeProxy):
                return self._value != other._value
            return self._value != other

        def __lt__(self, other):
            if isinstance(other, BasicTypeProxy):
                return self._value < other._value
            return self._value < other

        def __del__(self):
            del self._value

    return BasicTypeProxy(value)


if __name__ == "__main__":
    proxy_int = basic_type_proxy(5)
    print(f"{isinstance(proxy_int, int)}")  # out: True
    proxy_float = basic_type_proxy(3.14)
    proxy_int.copy_(6)
    print(proxy_int + 3)  # out: 8
    print(proxy_float - 1.14)  # out: 2.0
    print(proxy_int * 2)  # out: 10
    print(proxy_float / 2)  # out: 1.57

    proxy_bool = basic_type_proxy(True)

    def check_bool(b):
        if b:
            print("True")
        else:
            print("False")

    check_bool(proxy_bool)  # out: True
    proxy_bool.copy_(False)
    check_bool(proxy_bool)  # out: False
