"""
base classes for solutions to models
"""

from __future__ import annotations
from collections import namedtuple



class ModelSolution(object):

    def __init__(self, model: Model):
        self.model = model

    def solve(self, *args, **kwargs):
        return self

    def __repr__(self):
        class_, model_name = self.__class__.__name__, self.model.name
        if model_name:
            return f"{class_}({repr(model_name)})"
        return class_



class LinearSolution(ModelSolution):

    order = 0
    __arrays__ = ()

    def __init__(self, model: Model):
        super().__init__(model)
        self._array_creator = namedtuple('SolutionArrays', self.__arrays__)

    def __repr__(self):
        return f"LinearSolution(order={self.order})"

    @property
    def arrays(self):
        _arrs = tuple([getattr(self, a) for a in self.__arrays__])
        return self._array_creator(*_arrs)
