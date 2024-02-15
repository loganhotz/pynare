"""
module for translating between model-implied variables and observed ones

as of the morning of March 19th, we assume that the model is perfectly observed,
and state innovations are N(0, I). that is, the state space model is

    s_t = T s_{t-1} + R u_t    u_t ~ N(0, I)
    y_t = Z s_t
"""
from __future__ import annotations

import copy
import numpy as np

from pynare.core.variables import _create_endog_vars
from pynare.core.generic import DynamicModel
from pynare.core.solve.first_order import FirstOrderSolution

# from pynare.parsing.lexer import DynareLexer
# from pynare.parsing.parser import DynareParser

import pandas as pd
from rich import print
def aprint(arr): print(pd.DataFrame(arr))



class ObservedModel(DynamicModel):
    """
    this assumes that observed variables are not functions of exog vars
    """

    def __init__(
        self,
        model: Model,
        exprs: dict[str, str] = {}
    ):
        # make 'exprs' an optional kwarg just so a more explicit error can be used
        if not exprs:
            raise ValueError("ObservedModel requires observation equations")

        # add the names of the observed vars to the VarArray of endog vars
        self.obs = _create_endog_vars(list(exprs.keys()))
        endog = np.concatenate((model.endog, self.obs))

        # augment the existing model with the given mappings from model vars and
        #   params to observed variables
        model_exprs = copy.deepcopy(model.exprs)
        observation_exprs = [' = '.join((k, v)) for k, v in exprs.items()]
        model_exprs.extend(observation_exprs)

        self.local_params = model.local_params
        endog_names = model.endog.names
        super().__init__(
            endog=endog,
            stoch=model.stoch,
            determ=model.determ,
            params=model.params,
            shocks=model._shocks,
            exprs=model_exprs,
            initial={eg: i for eg, i in zip(endog_names, model.initial)},
            terminal={eg: i for eg, i in zip(endog_names, model.terminal)},
            historical={eg: i for eg, i in zip(endog_names, model.historical)},
            language=model.language,
            name=model.name
        )

    @property
    def solution(self):
        return ObservedSolution(self).solve()



class ObservedSolution(FirstOrderSolution):

    __arrays__ = ('T', 'R', 'Z')

    def solve(self):

        if self.model.name != 'herbst_schorfheide':
            raise NotImplementedError()

        # set 'gy' and 'gu' arrays
        _ = super().solve()

        # dynare excludes the 'pi' endogenous variable when constructing the Kalman
        #   Filter state space? perhaps because it is purely forward-looking?
        obs_idx = range(7)
        print(self.model.obs.names)
        _obs_idx = self.model.endog.get_loc(self.model.obs.names)
        print('_obs_idx =', _obs_idx)
        t_idx = self.model.indexes.state

        # print(self.gy)
        # print(self.gu)
        print('out of ObservedSolution "solve" method')

        # observations are subsets of the full first solution
        Z_tm1, self.T, self.R = self.gy[obs_idx], self.gy[t_idx], self.gu[t_idx]

        # columns of Z need to be translated from time (t-1) state vars to time t
        self.Z = np.matmul(Z_tm1, self.T)

        return self





class Observation(object):

    def __init__(self):
        pass



class ObservedExprs(object):

    def __init__(
        self,
        exprs: dict[str, str],
        endog: VarArray,
        params: MutableMapping,
        stoch: VarArray = []
    ):

        """
        self.endog = endog
        self.stoch = stoch

        self.params = params
        """

        # the names of the observed vars, and their relation to model vars
        self.obs = list(exprs.keys())
        self.measure = [ObservationParser(v).parse() for v in exprs.values()]

        print(exprs)

        for m in self.measure:
            m.describe()



"""
class ObservationParser(DynareParser):

    def __init__(
        self,
        expr_string
    ):
        lexer = DynareLexer(expr_string)
        super().__init__(lexer)


    def parse(self):
        return self.mexpr()
"""
