from functools import wraps


def freeze(cls, msg="Class {cls} is frozen. Cannot set {key} = {value}"):
    """Wrap and freeze a class so that a=A(); a.foo=4 fails if .foo isn't defined by the class """
    cls.__frozen = False

    def frozensetattr(self, key, value):
        if self.__frozen and not hasattr(self, key):
            raise AttributeError(msg.format(cls=cls.__name__, key=key, value=value))
        else:
            object.__setattr__(self, key, value)

    def init_decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            func(self, *args, **kwargs)
            self.__frozen = True
        return wrapper

    cls.__setattr__ = frozensetattr
    cls.__init__ = init_decorator(cls.__init__)

    return cls
