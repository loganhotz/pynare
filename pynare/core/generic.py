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
from pynare.core.solve import StateSpace

from pynare.utils.dtypes import is_iterable_not_str



class GenericModel(object):

    def __init__(
        self,
        endog: list,
        stoch: list,
        determ: list,
        params: dict,
        shocks: dict,
        exprs: list,
        local_params: dict,
        initial: dict,
        lead_lag: dict,
        terminal: dict,
        historical: dict,
        language: str,
        name: str
    ):
        # language model was defined in, an optional name for the model
        self.language = language
        self.name = name

        if endog:
            _endog = _create_endog_vars(
                names=endog,
                initial=initial,
                lead_lag=lead_lag,
                terminal=terminal,
                historical=historical
            )
            self.endog = VarArray(_endog)
        else:
            self.endog = VarArray()

        if stoch:
            _stoch = _create_stoch_vars(names=stoch, shocks=shocks)
            self.stoch = VarArray(_stoch)
        else:
            self.stoch = VarArray()

        if determ:
            _determ = _create_determ_vars(names=determ, shocks=shocks)
            # ensures `determ` attribute will be a list regardless of no. of determ
            self.determ = VarArray()
        else:
            self.determ = VarArray()

        # dictionaries of parameters; of the form `name: value`
        self.params = ModelCalibration(params)
        self.local_params = ModelCalibration(local_params)

        # list of ASTs or string expressions. set the `lead_lag` attribute of the
        #    endogenous variables that were passed through the `endog` parameter
        self.exprs = ModelExprs(exprs)
        self.endog = set_lead_lags(self.endog, self.exprs)

        # don't particularly like this implementation but can't think of a way
        #    around it yet
        self.is_altered = False

        # prepare structural features of the model that are accessed via attributes
        #   of the same name, without the leading underscore
        self._indexes = VariableIndexManager(self.endog, self.stoch)

        # don't particularly like passing a partially initialized model...
        self._steady_state_repr = SteadyRepr(self)
        self._dynamic_repr = DynamicRepr(self)
        self._statespace = StateSpace(self)


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
    def steady_state_repr(self):
        """
        prepare the steady state representation of the model based on the ASTs in the
        model's ModelExprs
        """
        if self.is_altered:
            self.update()

        return self._steady_state_repr


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
    def initial(self):
        """ initial values of the endogenous parameters """
        return np.array([eg.initial for eg in self.endog], dtype=float)


    @property
    def terminal(self):
        """ terminal values of the endogenous parameters """
        return np.array([eg.terminal for eg in self.endog], dtype=float)


    @property
    def historical(self):
        """ historical values of the endogenous parameters """
        return np.array([eg.historical for eg in self.endog], dtype=float)


    @property
    def model_scope(self):
        return {**self.params, **self.local_params}


    @property
    def exog(self):
        return self.stoch + self.determ


    @property
    def indexes(self):
        if self.is_altered:
            self.update()

        return self._indexes


    @property
    def statespace(self):
        if self.is_altered:
            self.update()

        return self._statespace


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
                j = stoch_names.index(shk.other)
                cov = np.sqrt(shk.variance*sig[j, j]) * shk.corr

                sig[i, j], sig[j, i] = cov, cov

            # with an `if` here, any correlated variables that are declared as
            #    covarying ones later in the shock block are overwritten
            if isinstance(shk.cov, (int, float)):
                j, cov = stoch_names.index(shk.other), shk.cov
                sig[i, j], sig[j, i] = cov, cov

        return sig


    def update(self, *args, **kwargs):
        self.is_altered = False

        self.endog = set_lead_lags(self.endog, self.exprs)
        self._sigma = self._define_sigma()

        # structural features. steady, dynamic, statespace order is important,
        #   which I'm not particularly fond of...
        self._indexes = VariableIndexManager(self.endog, self.stoch)
        self._steady_state_repr = SteadyRepr(self)
        self._dynamic_repr = DynamicRepr(self)
        self._statespace = StateSpace(self)


    def __repr__(self):
        if self.name:
            return f"Model('{self.name}')"
        else:
            n_endog = len(self.endog)
            n_exog = len(self.exog)
            return f"Model({n_endog} endog, {n_exog} exog)"


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
            local_params=copy.deepcopy(self.local_params),
            initial=init,
            lead_lag=ll,
            terminal=term,
            historical=hist,
            language=self.language,
            name=self.name
        )
