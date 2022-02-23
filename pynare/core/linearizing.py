"""
module for computing Jacobian & higher-order derivative matrices of the model
"""

from __future__ import annotations
from functools import cached_property
from autograd import grad, jacobian

import warnings
import numpy as np



class Linearization(object):

    def __init__(
        self,
        model: Model
    ):
        # save pointer to model so higher-order linearizations can access already-
        #    computed linearized funcs of lower-order linearizations
        self.model = model

        # to-be-filled list of functions representing the steady-state or dynamic model
        self.linearized_funcs = []
        self.funcs = []


    def differentiate_functions(self):
        # default to SteadyJacobian and ModelJacobian method
        for func in self.funcs:
            dfunc = grad(func)
            self.linearized_funcs.append(dfunc)


    @cached_property
    def array(self):
        # default to Jacobian method here, too. differentiate functions if, somehow,
        #   they're not already created
        if not self.linearized_funcs:
            self.differentiate_functions()

        n_exprs, n_cols = len(self.linearized_funcs), len(self.ss)
        jac = np.zeros((n_exprs, n_cols))

        for i, func in enumerate(self.linearized_funcs):
            jac[i, :] = np.array(func(self.ss))

        return jac


    @cached_property
    def ss(self):
        # default to dynamic steady state that includes stochastic shocks
        llx = self.model.indexes.llx

        # order endog variables in (t-1, t, t+1) declaration order
        ss_block = np.tile(self.model.ss.values, (3, 1))
        ss_endog = ss_block[~np.isnan(llx)].flatten()

        # all stochastic shocks are zero in steady state
        zero_stochs = np.zeros(len(self.model.stoch))
        return np.concatenate((ss_endog, zero_stochs))


    def __repr__(self):
        n_exprs = len(self.funcs)
        n_ss = len(self.ss)
        return f"{self.__class__.__name__}({n_exprs} exprs, {n_ss} steady values)"



class SteadyJacobian(Linearization):

    def __init__(
        self,
        model: Model
    ):
        super().__init__(model)

        # list of functions representing the steady-state model
        self.funcs = self.model.steady_state_repr.func_list
        self.differentiate_functions()


    @property
    def ss(self):
        return self.model.ss.values

    def __repr__(self):
        n_exprs = len(self.funcs)
        return f"{self.__class__.__name__}({n_exprs} exprs)"



class ModelJacobian(Linearization):

    def __init__(
        self,
        model: Model
    ):
        super().__init__(model)

        # list of functions representing the dynamic model
        self.funcs = self.model.dynamic_repr.func_list
        self.differentiate_functions()



class ModelHessian(Linearization):

    def __init__(
        self,
        model: Model
    ):
        super().__init__(model)

        # list of functions representing the dynamic model
        self.funcs = self.model.dynamic_repr.func_list
        self.differentiate_functions()

    def differentiate_functions(self):
        model_jac = self.model.jacobian
        for func in model_jac.linearized_funcs:
            hess = jacobian(func)
            self.linearized_funcs.append(hess)

    @cached_property
    def array(self):
        # differentiate functions if, somehow, they're not already created
        if not self.linearized_funcs:
            self.differentiate_functions()

        n_exprs, n_cols = len(self.linearized_funcs), len(self.ss) ** 2
        hess = np.zeros((n_exprs, n_cols))

        for i, func in enumerate(self.linearized_funcs):
            with warnings.catch_warnings():
                # when a function whose hessian is to be evaluated will have an
                #    all-zero hessian, autograd throws some UserWarnings. for
                #    effiency, in the future, we'll check for that first & just
                #    return a vector of 0s
                warnings.simplefilter('ignore')
                hess[i, :] = np.array(func(self.ss)).flatten('C')

        return hess
