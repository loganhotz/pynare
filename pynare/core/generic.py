from __future__ import annotations

import copy
import numpy as np

from pynare.core.expressions import (
    ModelExprs,
    SteadyRepr,
    DynamicRepr,
)
from pynare.core.variables import (
    VarArray,
    set_lead_lags,
    _create_endog_vars,
    _create_stoch_vars,
    _create_determ_vars,
    VariableIndexManager,
    ModelCalibration
)
from pynare.core.steady import SteadyState
from pynare.core.solve import StateSpace

from pynare.utils.dtypes import is_iterable_not_str



class ModelSkeleton(object):
    """
    the bare minimum for a model:
        endog    : endogenous vars whose values affect one another
        params   : parameter values that determine behavior of 'endog' vars
        language : the language the model was written in
        name     : an optional name for the model used in I/O processes
    """

    def __init__(
        self,
        endog: list,
        params: dict,
        language: str,
        name: str = ''
    ):
        if language != 'dynare':
            raise ValueError(f"'{language}'. unrecognized language")

        # language the model was defined in; an optional name for the model
        self.language = language
        self.name = name

        # accept empty lists in case `md = pynare.Model()` is used
        if np.count_nonzero(endog):
            self.endog = _create_endog_vars(names=endog)

        else:
            self.endog = VarArray()

        # dictionaries of parameters; of the form `name: value`
        self.params = ModelCalibration(params)

        try:
            self.model_scope = self.params
        except AttributeError:
            # subclass has 'model_scope' as an attribute
            pass

        # don't particularly like this implementation but can't think of a way
        #    around it yet
        self.is_altered = False

    def __repr__(self):
        class_ = self.__class__.__name__
        if self.name:
            return f"{class_}('{self.name}')"
        else:
            return f"{class_}({len(self.endog)} endog)"

    def add_endog(self, endog, **kwargs):
        """
        add one or more EndogVar objects to the `endog` attribute

        Parameters
        ----------
        endog : str | Iterable[str]
            the name(s) of the EndogVar(s) to be created
        kwargs : keyword arguments
            attributes of the EndogVar to set. possible keys are 'initial', 'lead_lag',
            'terminal', and 'historical'

        Returns
        -------
        None
        """
        _endog = _create_endog_vars(endog, **kwargs)
        self.endog = np.concatenate((self.endog, _endog))

        self.is_altered = True

    def add_params(self, **kwargs):
        """
        add a parameter to the model. no checks are made if parameter already exists,
        so this can be used to change the value of existing parameters, too

        Parameters
        ----------
        kwargs : keyword arguments
            each (key, value) pair is the parameter name and value

        Returns
        -------
        None
        """
        for k, v in kwargs.items():
            self.params[k] = v

        self.is_altered = True

    def update(self):
        """
        nothing to change here
        """
        self.is_altered = False



class StructuredModel(ModelSkeleton):
    """
    adding relationships between endog vars
        exprs       : the parse-able model expressions that show how endog vars are
                      linked and how they depend on params
        steady_repr : the steady-state representation of the 'exprs'. assumes that
                      exog vars do not shock the endog ones
        ss          : the steady-state object of the model
    """

    def __init__(
        self,
        endog: list,
        params: dict,
        exprs: list,
        language: str,
        name: str = ''
    ):
        super().__init__(
            endog=endog,
            params=params,
            language=language,
            name=name
        )

        # list of ASTs or string expressions. set 'lead_lag' of endog vars
        self.exprs = ModelExprs(exprs)
        self.endog = set_lead_lags(self.endog, self.exprs)

        # i don't particularly like passing a partially initialized model...
        self._steady_repr = SteadyRepr(self)

    def add_exprs(self, exprs):
        """
        add a single, or more than one, expression to the model definition

        Parameters
        ----------
        exprs : str | AST | Iterable[str, AST]
            the expressions to add to the model

        Returns
        -------
        None
        """
        if is_iterable_not_str(exprs):
            self.exprs.extend(exprs)
        else:
            self.exprs.append(exprs)

        self.is_altered = True

    @property
    def steady_repr(self):
        """
        prepare the steady state representation of the model based on the ASTs in the
        model's ModelExprs
        """
        if self.is_altered:
            self.update()

        return self._steady_repr

    @property
    def ss(self):
        """
        returns the model's SteadyState object. re-evaluated each time this attribute
        is accessed
        """
        return SteadyState(self)

    @property
    def initial(self):
        """
        initial values of the endogenous parameters. needed to initialize the steady-
        state computation process, and can be overwritten by subclasses
        """
        return np.array([eg.initial for eg in self.endog], dtype=float)

    def update(self, *args, **kwargs):
        self.is_altered = False

        self.endog = set_lead_lags(self.endog, self.exprs)
        self._steady_repr = SteadyRepr(self)



class DynamicModel(StructuredModel):
    """
    incorporating stochastic and deterministic exogenous variables into a model
    """

    def __init__(
        self,
        endog: list,
        params: dict,
        exprs: list,
        stoch: list,
        determ: list,
        shocks: dict,
        initial: dict,
        terminal: dict,
        historical: dict,
        language: str,
        name: str
    ):
        super().__init__(
            endog=endog,
            params=params,
            exprs=exprs,
            language=language,
            name=name
        )

        # 'lead_lag' property was set upon StructuredModel initialization
        self.endog = _create_endog_vars(
            names=self.endog.names,
            lead_lag={eg.name: eg.lead_lag for eg in self.endog},
            initial=initial,
            terminal=terminal,
            historical=historical
        )

        # initialize exogenous variables
        if np.count_nonzero(stoch):
            self.stoch = _create_stoch_vars(names=stoch, shocks=shocks)
        else:
            self.stoch = VarArray()

        if np.count_nonzero(determ):
            self.determ = _create_determ_vars(names=determ, shocks=shocks)
        else:
            self.determ = VarArray()

        # prepare dynamic structural features of the model that are accessed via
        #   attributes of the same name, without the leading underscore
        self._indexes = VariableIndexManager(self.endog, self.stoch)
        self._dynamic_repr = DynamicRepr(self)
        self._statespace = StateSpace(self)


    def add_stoch(self, stoch, **kwargs):
        """
        add one or more StochVar objects to the `stoch` attribute

        Parameters
        ----------
        stoch : str | Iterable[str]
            the name(s) of the StochVar(s) to be created
        kwargs : keyword arguments
            attributes of the StochVar to set. possible keys are 'other', 'variance',
            'corr', and 'cov'

        Returns
        -------
        None
        """
        _stoch = _create_stoch_vars(stoch, **kwargs)
        self.stoch = np.concatenate((self.stoch, _stoch))

        self.is_altered = True

    def add_determ(self, determ, **kwargs):
        """
        add one or more DetermVar objects to the `determ` attribute

        Parameters
        ----------
        determ : str | Iterable[str]
            the name(s) of the DetermVar(s) to be created
        kwargs : keyword arguments
            attributes of the DetermVar to set. possible keys are 'periods',
            and 'values'

        Returns
        -------
        None
        """
        _determ = _create_determ_vars(determ, **kwargs)
        self.determ = np.concatenate((self.determ, _determ))

        self.is_altered = True


    @property
    def terminal(self):
        """ terminal values of the endogenous parameters """
        return np.array([eg.terminal for eg in self.endog], dtype=float)

    @property
    def historical(self):
        """ historical values of the endogenous parameters """
        return np.array([eg.historical for eg in self.endog], dtype=float)

    @property
    def exog(self):
        return np.concatenate((self.stoch, self.determ))


    # structural properties of a dynamic model
    @property
    def indexes(self):
        if self.is_altered:
            self.update()

        return self._indexes

    @property
    def dynamic_repr(self):
        """
        prepare the dynamic representation of the model based on the ASTs in the
        model's ModelExprs
        """
        if self.is_altered:
            self.update()

        return self._dynamic_repr

    @property
    def statespace(self):
        if self.is_altered:
            self.update()

        return self._statespace


    # dynamic array managers
    @property
    def sigma(self):
        """returns the covariance matrix of the stochastic shocks"""
        if self.is_altered or not hasattr(self, '_sigma'):
            self.update()
        return self._sigma

    @property
    def jacobian(self):
        """returns the dynamic jacobian of the model"""
        return self.dynamic_repr.jacobian

    @property
    def hessian(self):
        """returns the dynamic hessian of the model"""
        return self.dynamic_repr.hessian


    # VarArray-returning properties
    @property
    def state_vars(self): return self.endog[self.indexes.state]

    @property
    def pure_forward_vars(self): return self.endog[self.indexes.pure_forward]

    @property
    def pure_backward_vars(self): return self.endog[self.indexes.pure_backward]

    @property
    def mixed_vars(self): return self.endog[self.indexes.mixed]

    @property
    def static_vars(self): return self.endog[self.indexes.static]

    @property
    def forward_vars(self): return self.endog[self.indexes.forward]

    @property
    def backward_vars(self): return self.endog[self.indexes.backward]


    def _define_sigma(self):
        """
        define the covariance matrix for the stochastic shocks
        """

        # pre-allocate
        stoch_names = self.stoch.names
        n_stoch = len(self.stoch)
        sig = np.zeros((n_stoch, n_stoch), dtype=float)

        # first initialize variance, in case a correlated variable is given
        #    and needs to be rescaled to covariance
        for i, shk in enumerate(self.stoch):
            if isinstance(shk.variance, (int, float)):
                sig[i, i] = shk.variance

        for i, shk in enumerate(self.stoch):
            if isinstance(shk.corr, (int, float)):
                j = np.where(stoch_names == shk.other)[0]
                cov = np.sqrt(shk.variance*sig[j, j]) * shk.corr

                sig[i, j], sig[j, i] = cov, cov

            # with an `if` here, any correlated variables that are declared as
            #    covarying ones later in the shock block are overwritten
            if isinstance(shk.cov, (int, float)):
                j, cov = np.where(stoch_names == shk.other)[0], shk.cov
                sig[i, j], sig[j, i] = cov, cov

        return sig


    def update(self, *args, **kwargs):
        self.is_altered = False

        self.endog = set_lead_lags(self.endog, self.exprs)
        self._sigma = self._define_sigma()

        # structural features. steady, dynamic, statespace order is important,
        #   which I'm not particularly fond of...
        self._indexes = VariableIndexManager(self.endog, self.stoch)
        self._steady_repr = SteadyRepr(self)
        self._dynamic_repr = DynamicRepr(self)
        self._statespace = StateSpace(self)


    def copy(self):

        # extract relevant information for endogenous variables
        endog, ll, init, term, hist = [], {}, {}, {}, {}
        for eg in self.endog:
            name, l = eg.name, eg.lead_lag
            i, t, h = eg.initial, eg.terminal, eg.historical

            endog.append(name)

            if l is not None: ll[name] = l
            if i is not None: init[name] = i
            if t is not None: term[name] = t
            if h is not None: hist[name] = h

        # extract relevant information for stochs
        stoch, shocks = [], {}
        for sc in self.stoch:
            name, o = sc.name, sc.other
            v, r, c = sc.variance, sc.corr, sc.cov

            stoch.append(name)
            sc_dict = {}

            if o is not None: sc_dict['other'] = o
            if v is not None: sc_dict['variance'] = v
            if r is not None: sc_dict['corr'] = r
            if c is not None: sc_dict['cov'] = c

            shocks[name] = sc_dict

        # extract relevant information for determs
        determ = []
        for dt in self.determ:
            name, p, v = dt.name, dt.periods, dt.values

            determ.append(name)
            dt_dict = {}

            if p is not None: dt_dict['periods'] = p
            if v is not None: dt_dict['values'] = v

            shocks[name] = dt_dict

        class_ = self.__class__
        return class_(
            endog=endog,
            stoch=stoch,
            determ=determ,
            params=copy.deepcopy(self.params),
            shocks=shocks,
            exprs=copy.deepcopy(self.exprs),
            initial=init,
            terminal=term,
            historical=hist,
            language=self.language,
            name=self.name
        )
