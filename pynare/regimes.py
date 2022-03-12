"""
module for regime-switching models
"""
from __future__ import annotations



class RegimeSwitchingModel(object):
    """
    a model that moves between structurally different regimes

    model : Model
        the baseline model whose expressions and/or `sigma` matrix will differ between
        the provided regimes
    exprs : dict | Sequence[dict]
        a single dict of the form {eq_number: model expression}, or an iterable of
        such dicts. each dict dictates how each alternative regime differs from the
        baseline model.
            if a single dict is passed, then the model will switch between two regimes,
        the baseline one passed through the `model` parameter, and the one defined by
        the `exprs` dict. if n > 1 dicts are passed, then the model switches between
        n + 1 regimes
    sigma : dict
        the dictionary of shock expressions defining a new regime
    """

    def __init__(
        self,
        model: Model,
        exprs: dict | Sequence[dict] = {},
        sigma: dict | Sequence[dict] = {}
    ):
        if sigma:
            raise NotImplementedError("sigma")

        if not (exprs or sigma):
            raise ValueError("one of 'exprs' or 'sigma' needs to be specified")

        # hacky -- need to find a way to retrieve attributes of the baseline model
        #   if it's not going to be treated as such
        self.name = model.name

        self.exprs = exprs

        if isinstance(self.exprs, dict):
            mc = model.copy()
            for eq, expr in self.exprs.items():
                mc.exprs[eq] = expr

            mc.is_altered = True
            self.models = [model, mc]

        else:
            self.models = [model.copy()]
            for expr in self.exprs:
                mc = model.copy()
                mc.is_altered = True

                for eq, ex in expr.items():
                    mc.exprs[eq] = ex

                self.models.append(mc)

    def __getitem__(self, key: int):
        return self.models[key]

    def __len__(self):
        return len(self.exprs) + 1

    def __repr__(self):
        return f"RegimeSwitchingModel(n={len(self)})"
