"""
wrapping matplotlib.pyplot, so imports of the kind

    >>> import pynare.plotting as plt

can be used instead
"""
import matplotlib.pyplot as _plt

_all_ = []



def wrap_namespace(old, new):

    for name, obj in old.items():
        _all_.append(name)
        new[name] = obj


wrap_namespace(_plt.__dict__, globals())
__all__ = _all_
