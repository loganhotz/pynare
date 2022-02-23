from collections import abc
from numbers import Number

import numpy as np



def is_iterable_not_str(obj) -> bool:
    """
    check if an object is a non-string iterable

    Parameters
    ----------
    obj : object
        object to check

    Returns
    -------
    bool
    """
    return isinstance(obj, abc.Iterable) and not isinstance(obj, str)


def is_number(obj) -> bool:
    """
    check if an object is a number. booleans will return True because they are
    `int` subclasses

    Parameters
    ----------
    obj : object
        object to check

    Returns
    -------
    bool
    """
    return isinstance(obj, (Number, np.number))


def ensure_list(obj) -> list:
    """
    ensure an object is a list

    Parameters
    ----------
    obj : object
        object to possibly wrap

    Returns
    -------
    list
    """
    if is_iterable_not_str(obj):
        return list(obj)

    return [obj]
