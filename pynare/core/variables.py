from __future__ import annotations

from functools import cached_property
from collections import Sequence
from collections.abc import MutableMapping
import numpy as np

from pynare.errors import (
    EndogIndexError,
    CalibrationError
)

from pynare.utils.dtypes import is_iterable_not_str

from pynare.parsing import parse_string
import pynare.parsing.base as base
import pynare.parsing.ast as ast



class VarArray(np.ndarray):

    def __new__(cls, variables: Iterable = [], *args, **kwargs):
        # input array is a list of ModelVar instances
        obj = np.asarray(variables).view(cls)

        return obj

    def __array_finalize__(self, obj):
        try:
            names = [v.name for v in self]
        except AttributeError:
            raise TypeError("all variables must be a subclass of ModelVar")

        try:
            vtypes = [v.vtype for v in self]
        except AttributeError:
            raise TypeError("all variables must be a subclass of ModelVar")

        # ensure all variables are of the same type
        try:
            first = vtypes[0]
            if vtypes.count(first) != len(vtypes):
                raise TypeError("variables were not all of the same vtype")

        except IndexError:
            first = ''

        self.vtype = first
        self.names = np.array(names, dtype=str)

    def __array_function__(self, func, types, args, kwargs):
        """
        this method is called by top-level numpy methods. it might have to be extended
        at some point, but for the moment, the collection of variables in VarArray just
        needs to be maleable
        """
        if func == np.concatenate:

            var_seq, variables = args[0], []
            for var in var_seq:
                if isinstance(var, VarArray) or is_iterable_not_str(var):
                    variables.extend(var)
                else:
                    variables.append(var)

            return VarArray(variables)

        else:
            return NotImplemented

    def __contains__(self, obj):
        if isinstance(obj, str):
            return obj in self.names
        if isinstance(obj, ModelVar):
            return obj.name in self.names
        return False

    def __repr__(self):
        return f"VarArray(vtype={self.vtype})"

    def __str__(self):
        return f"VarArray([{', '.join(self.names)}], vtype={self.vtype})"

    def get_loc(self, key: str | Iterable[str]):
        """
        retrieve the locations of variables in VarArray based one their name

        Parameters
        ----------
        key : str | Iterable[str]
            the desired variable names

        Returns
        -------
        numpy array of integer locations

        Notes
        -----
        we choose not to use `np.where(np.in1d(...))` because the resulting index
        array sorts its elements based on their locations in `key`, instead of their
        ordering in `self.names`. see
            https://stackoverflow.com/questions/8251541/numpy-for-every-element-in-
            one-array-find-the-index-in-another-array
        """
        from rich import print

        if is_iterable_not_str(key):
            idx = np.zeros(len(key), dtype=int)

            for i, var in enumerate(key):
                name = var.name if isinstance(var, ModelVar) else var
                loc = np.where(name == self.names)[0]
                if not loc.size:
                    raise KeyError(f"{repr(var)} is not a variable")
                idx[i] = loc

            return idx

        else:
            name = key.name if isinstance(key, ModelVar) else key
            loc = np.where(name == self.names)[0]
            if not loc.size:
                raise KeyError(f"{repr(key)} is not a variable")

            return loc.item()



class ModelVar(object):

    def __init__(self, name: str, vtype: str = ''):
        if not isinstance(name, str):
            type_ = type(name).__name__
            raise TypeError(f"'{type_}'. model variable name must be str.")
        self.name = name
        self.vtype = vtype

    def __repr__(self):
        class_ = self.__class__.__name__
        return f"{class_}({self.name})"

    def __str__(self):
        return self.name



endog_bounds = ('initial', 'terminal', 'historical')
lead_lag_default = None
initial_default = None
terminal_default = None
historical_default = None

class EndogVar(ModelVar):

    def __init__(
        self,
        name: str,
        initial: Union[Any, float] = initial_default,
        lead_lag: Union[Any, Iterable[int]] = lead_lag_default,
        terminal: Union[Any, float] = terminal_default,
        historical: Union[Any, float] = historical_default
    ):
        super().__init__(name, 'endog')

        self.lead_lag = lead_lag

        # if all three boundary values are not provided, default to initial
        #    value of 0
        if (terminal is None) and (historical is None):
            self.initial = initial if initial else 0
        else:
            self.initial = initial

        self.terminal = terminal
        self.historical = historical

    def set_lead_lag(self, ll):
        self.lead_lag = ll

    def __repr__(self):
        bounds_dict = {
            b: getattr(self, b) for b in endog_bounds
            if getattr(self, b) is not None
        }
        bounds = ', '.join(f"{n}={v}" for n, v in bounds_dict.items())
        return f"EndogVar({self.name}, {bounds})"



class ExogVar(ModelVar):

    def __init__(self, name: str, vtype: str = 'exog'):
        super().__init__(name, vtype)



class StochVar(ExogVar):

    def __init__(
        self,
        name: str,
        other: str = '',
        variance: Union[int, float] = None,
        corr: Union[int, float] = None,
        cov: Union[int, float] = None
    ):
        super().__init__(name, 'stoch')
        self.other = other

        self.variance = variance
        self.corr = corr
        self.cov = cov

    def __repr__(self):
        name = self.name
        if self.other:
            var, other = self.variance, self.other
            if self.corr:
                c, cn = 'corr', self.corr
            else:
                c, cn = 'cov', self.cov
            return f"StochVar({name}, variance={var}, {c}({name}, {other})={cn})"
        else:
            return f"StochVar({name}, variance={self.variance})"



class DetermVar(ExogVar):

    def __init__(
        self,
        name: str,
        periods: Iterable[int] = None,
        values: Iterable[float] = None
    ):
        super().__init__(name, 'determ')

        self.periods = periods
        self.values = values

    def __repr__(self):
        n, p, v = self.name, self.periods, self.values
        return f"DetermVar({n}, periods={p}, values={v})"



def _create_endog_vars(
    names: Union[Iterable[str], str],
    initial: Union[Dict, float] = {},
    lead_lag: Dict = {},
    terminal: Union[Dict, float] = {},
    historical: Union[Dict, float] = {}
):
    """
    create a single or list of EndogVar objects

    Parameters
    ----------
    names : str | Iterable[str]
        the name(s) of the EndogVar(s) to be created
    initial, terminal, historical : dict | float
        the boundary values of the endogenous variables in the model. if a dict,
        this is assumed to be of the form `name: value`. if a float, this is the
        boundary value used for all the passed endogenous variable names
    lead_lag : dict
        the record of lead & lag incidences of the endogenous variables in the model

    Returns
    -------
    list of EndogVar instances
    """

    def _create_endog(name):
        # bunch of `try ... Except ...` blocks for dict/scalar checks
        try:
            i = initial.get(name, initial_default)
        except AttributeError:
            i = initial

        try:
            ll = lead_lag.get(name, lead_lag_default)
        except AttributeError:
            ll = lead_lag

        try:
            t = terminal.get(name, terminal_default)
        except AttributeError:
            t = terminal

        try:
            h = historical.get(name, historical_default)
        except AttributeError:
            h = historical

        return EndogVar(name, i, ll, t, h)


    if is_iterable_not_str(names):
        # many endogenous variables
        return [_create_endog(name) for name in names]

    else:
        # single endogenous variable
        return [_create_endog(names)]



def _create_stoch_vars(
    names: Union[str, Iterable[str]],
    shocks: dict = {}
):
    """
    create a single or list of StochVar objects

    Parameters
    ----------
    names : str | Iterable[str]
        the name(s) of the StochVar(s) to be created
    shocks : dict ( = {} )
        dict of the form `name: moments`, where `moments` is another dictionary,
        with possible entries being:
            `other`    : the covarying/correlated variable
            `variance` : the variance of the shock `name`
            `corr`     : the correlation of `name` and `other`
            `cov`      : the covariance of `name` and `other`

    Returns
    -------
    list of StochVar instances
    """
    if not isinstance(shocks, dict):
        type_ = type(shocks).__name__
        raise TypeError(f"{type_}. shock definitions must be a dict")

    default_dict = {'variance': 1.0}
    def _create_stoch(name):
        shock_dict = shocks.get(name, default_dict)
        return StochVar(name, **shock_dict)

    if is_iterable_not_str(names):
        # many stochastic shocks
        return [_create_stoch(name) for name in names]

    else:
        # single stochastic shock
        return [_create_stoch(names)]



def _create_determ_vars(
    names: Union[str, Iterable[str]],
    shocks: dict = {}
):
    """
    create a single or list of DetermVar objects

    Parameters
    ----------
    names : str | Iterable[str]
        the name(s) of the DetermVar(s) to be created
    shocks : dict ( = {} )
        dict of the form `name: moments`, where `moments` is another dictionary,
        with possible entries being:
            `periods` : the periods where the disturbances in `values` are realized
            `values`  : the values of disturbances to the exogenous variables

    Returns
    -------
    list of DetermVar instances
    """
    if not isinstance(shocks, dict):
        type_ = type(shocks).__name__
        raise TypeError(f"{type_}. shock definitions must be a dict")

    default_dict = {'periods': 0, 'values': 1}
    def _create_determ(name):
        determ_dict = shocks.get(name, default_dict)
        return DetermVar(name, **determ_dict)

    if is_iterable_not_str(names):
        # many deterministic shocks
        return [_create_determ(name) for name in names]

    else:
        # single deterministic shock
        return [_create_determ(names)]


def set_lead_lags(
    endog: Iterable[EndogVar],
    exprs: Union[Iterable[AST], ModelExprs]
):
    """
    sets the `lead_lag` attribute of a collection of EndogVar objects. this
    is done in-place

    Parameters
    ----------
    endog : Iterable[EndogVar]
        the list of endogenous variables to alter
    exprs : Iterable[AST] | ModelExprs
        the model expressions that contains information about the periods that
        the endogenous variables appear in

    Returns
    -------
    endog_ll : list
        the same list of endogenous variables, with `lead_lag` set
    """
    var_periods = read_lead_lags(endog, exprs)
    for eg in endog:
        periods = var_periods[eg.name]
        eg.set_lead_lag(periods)

    return endog


def read_lead_lags(
    endog: Union[Iterable[ModelVar], Iterable[str]],
    exprs: Iterable[AST]
):
    """
    create a dict of the lead & lag periods of a model's endogenous variables

    Parameters
    ----------
    endog : Iterable[ModelVar] | Iterable[str]
        list of endogenous variables or endogenous variable names whose periods
        will be recorded
    exprs : Iterable[AST]
        an iterable of AST expressions to search over

    Returns
    -------
    lead_lags : dict
        a dictionary of the form {`name`: `set of periods`} that records the
        periods variable `name` appears in
    """

    try:
        endog_names = endog.names
        # endog_names = [eg.name for eg in endog]
    except AttributeError:
        endog_names = endog

    lead_lags = dict.fromkeys(endog_names, set())

    for ep in exprs:
        counter = LeadLagCounter(endog_names, ep)
        counts = counter.count()

        for var, periods in counts.items():
            lead_lags[var] = lead_lags[var].union(periods)

    return lead_lags



class LeadLagCounter(base.ABCVisitor):

    def __init__(self, endog: Iterable[str], tree: AST):
        super().__init__(tree)

        self.endog = endog
        self.endog_periods = {}

    def count(self):
        self.visit(self.tree)
        return self.endog_periods

    def visit_BinaryOp(self, node):
        self.visit(node.left)
        self.visit(node.right)

    def visit_UnaryOp(self, node):
        self.visit(node.expr)

    def visit_Num(self, node):
        pass

    def visit_Function(self, node):
        self.visit(node.expr)

    def visit_Var(self, node):
        var_name = node.value
        if var_name in self.endog:
            var_period = self.endog_periods.get(var_name, set())
            var_period.add(0)
            self.endog_periods[var_name] = var_period

    def visit_PeriodVar(self, node):
        var_name = node.var.value
        if var_name in self.endog:
            if node.direction.type == base.PLUS:
                period = int(node.offset.value)
            else:
                period = -int(node.offset.value)

            var_period = self.endog_periods.get(var_name, set())
            var_period.add(period)
            self.endog_periods[var_name] = var_period



class VariableIndexManager(object):
    """
    manager class for keeping track of indices of model variables in the Jacobian
    and Decision Rule spaces
    """

    def __init__(
        self,
        endog: Iterable[EndogVar],
        stoch: Iterable[StochVar]
    ):
        # dict of `name: periods`
        self.lead_lags = {eg.name: tuple(sorted(eg.lead_lag)) for eg in endog}

        # prepare the lead-lag incidence array by creating indicators for locations
        #    of endogenous variables at all periods
        llx = np.full((3, len(endog)), fill_value=np.nan)
        for idx, ll in enumerate(self.lead_lags.values()):
            lead_lag = np.array(list(ll)) + 1
            try:
                llx[lead_lag, idx] = 1
            except IndexError:
                if not lead_lag:
                    problem_endog = endog[idx]
                    raise EndogIndexError(
                        f"variable '{str(problem_endog)}' was declared as endog but "
                        "was not seen at any time period"
                    ) from None

        # number of endog variables incidences at all periods. then replace indicators
        #   with the jacobian column index
        self.n_endog_inc = int(np.nansum(llx))
        llx[llx == 1] = np.arange(self.n_endog_inc)
        self.llx = llx

        # dict of `name: declaration index`. record stochastic 
        self.var_cols = {eg.name: idx for idx, eg in enumerate(endog)}
        self.stoch = stoch
        self.n_exog = len(stoch)

        # save variable typers for dr and declaration order accesses
        self.dr_typer = VariableTyper(llx[:, self.dr_order])
        self.dc_typer = VariableTyper(llx)


    def __repr__(self):
        return f"VariableIndexManager({self.n_endog} endog, {self.n_exog} exog)"

    @cached_property
    def dr_order(self):
        llx = self.llx

        static_idx = np.where(np.isnan(llx[0]) & np.isnan(llx[2]))[0]

        not_forward = np.isnan(llx[1:]).any(axis=0)
        back_idx = np.where(~np.isnan(llx[0]) & not_forward)[0]

        mixed_idx = np.where(~np.isnan(llx[0]) & ~np.isnan(llx[2]))[0]
        pure_for_idx = np.where(np.isnan(llx[0]) & ~np.isnan(llx[2]))[0]

        return np.concatenate((static_idx, back_idx, mixed_idx, pure_for_idx))

    @cached_property
    def dc_order(self):
        return np.argsort(self.dr_order)

    @cached_property
    def n_static(self):
        llx = self.llx
        return np.count_nonzero(llx[1, np.isnan(llx[0]) & np.isnan(llx[2])])

    @cached_property
    def n_forward(self):
        return np.count_nonzero(~np.isnan(self.llx[2]))

    @cached_property
    def n_backward(self):
        return np.count_nonzero(~np.isnan(self.llx[0]))

    @cached_property
    def n_mixed(self):
        return np.count_nonzero(~np.isnan(self.llx[0]) & ~np.isnan(self.llx[2]))

    @cached_property
    def n_dynamic(self):
        return self.n_endog - self.n_static

    @cached_property
    def n_state(self):
        return np.count_nonzero(~np.isnan(self.llx[0]))

    @cached_property
    def n_endog(self):
        return self.llx.shape[1]

    @cached_property
    def exogenous_jacobian(self):
        # exogenous variables are the right-most columns of the Jacobian
        return np.arange(self.n_endog_inc, self.n_endog_inc+self.n_exog, dtype=int)


    def __getattr__(self, attr):
        """
        creates an interface for getting indexes of variables in the jacobian array,
        in both declaration and decision-rule order
        """

        if attr == 'dr':
            return self.dr_typer

        try:
            return getattr(self.dc_typer, attr)

        except AttributeError:
            class_ = self.__class__.__name__
            raise AttributeError(f"'{class_}' object has no attribute '{attr}'")



class VariableTyper(object):

    def __init__(self, llx):
        self.llx = llx

    @cached_property
    def state(self):
        return np.where(~np.isnan(self.llx[0]))[0]

    @cached_property
    def state_timed(self):
        return TimedLLX(self.llx, ~np.isnan(self.llx[0]))

    @cached_property
    def cont(self):
        return np.where(~np.isnan(self.llx[1]))[0]

    @cached_property
    def cont_timed(self):
        return TimedLLX(self.llx, ~np.isnan(self.llx[1]))

    @cached_property
    def static(self):
        return np.where(np.isnan(self.llx[0]) & np.isnan(self.llx[2]))[0]

    @cached_property
    def static_timed(self):
        return TimedLLX(self.llx, np.isnan(self.llx[0]) & np.isnan(self.llx[2]))

    @cached_property
    def dynamic(self):
        return np.where(~(np.isnan(self.llx[0]) & np.isnan(self.llx[2])))[0]

    @cached_property
    def dynamic_timed(self):
        return TimedLLX(self.llx, ~(np.isnan(self.llx[0]) & np.isnan(self.llx[2])))

    @cached_property
    def forward(self):
        return np.where(~np.isnan(self.llx[2]))[0]

    @cached_property
    def forward_timed(self):
        return TimedLLX(self.llx, ~np.isnan(self.llx[2]))

    @cached_property
    def pure_forward(self):
        return np.where(np.isnan(self.llx[0]) & ~np.isnan(self.llx[2]))[0]

    @cached_property
    def pure_forward_timed(self):
        return TimedLLX(self.llx, np.isnan(self.llx[0]) & ~np.isnan(self.llx[2]))

    @cached_property
    def backward(self):
        return np.where(~np.isnan(self.llx[0]))[0]

    @cached_property
    def backward_timed(self):
        return TimedLLX(self.llx, ~np.isnan(self.llx[0]))

    @cached_property
    def pure_backward(self):
        return np.where(~np.isnan(self.llx[0]) & np.isnan(self.llx[2]))[0]

    @cached_property
    def pure_backward_timed(self):
        return TimedLLX(self.llx, ~np.isnan(self.llx[0]) & np.isnan(self.llx[2]))

    @cached_property
    def mixed(self):
        return np.where(~np.isnan(self.llx[0]) & ~np.isnan(self.llx[2]))[0]

    @cached_property
    def mixed_timed(self):
        return TimedLLX(self.llx, ~np.isnan(self.llx[0]) & ~np.isnan(self.llx[2]))

    def flatten(self):
        return self.llx[~np.isnan(self.llx)].astype(int)

    def __repr__(self):
        return "VariableTyper"



class TimedLLX(object):
    """
    given a lead-lag incidence array and an array that selects a subset of the
    endogenous variables, create an object that, when indexed into, returns the
    selected endogenous variables in a model period

    >>> tllx = TimedLLX(llx, [array that selects dynamic endog])
    >>> tlxx['t-1'] # returns an array of indices of dynamic variables at time t-1
    >>> tllx[0]     # returns an array of dynamic variables at time t (empty)
    >>> tllx['+1']  # returns an array of indices of dynamic variables at time t+1

    accepted keys
    -------------
        't-1', -1, '-1': returns time t-1 indices
        't', 0, '0'    : returns time t indices
        't+1', 1, '+1' : returns time t+1 indices
    """

    __periods__ = {
        't-1': 0,
        't': 1,
        't+1': 2,
        -1: 0,
        0: 1,
        1: 2,
        '-1': 0,
        '0': 1,
        '+1': 2
    }

    def __init__(
        self,
        llx: np.ndarray,
        selection: np.ndarray
    ):
        self.llx = llx
        self.selection = selection

    def __getitem__(self, key: Union[int, str]):

        try:
            timed_key = self.__periods__[key]
        except KeyError:
            accept = list(self.__periods__.keys())
            raise ValueError(f"{repr(key)}. accepted periods are {accept}")

        selected = self.llx[timed_key, self.selection]
        return selected[~np.isnan(selected)].astype(int)



class ModelCalibration(MutableMapping):
    """
    managing parameter values and caching their equation definitions
    """

    def __init__(self, defs):
        self.defs = defs

        calib = {}
        for k, v in self.defs.items():
            if isinstance(v, float):
                calib[k] = v
            elif isinstance(v, ast.Num):
                calib[k] = v.value

        self.calib = solve_calibration(self.defs)

    def __getitem__(self, key):
        try:
            return self.calib[key]
        except KeyError:
            raise KeyError(f"'{key}' is not a parameter") from None

    def __setitem__(self, key, value):
        self.defs[key] = value
        self.calib = solve_calibration(self.defs) # inefficient

    def __delitem__(self, key):
        del self.defs[key]
        del self.calib[key]

    def __iter__(self):
        return iter(self.calib)

    def __len__(self):
        return len(self.calib)

    def __repr__(self):
        return f"ModelCalibration({len(self)} params)"

    def __str__(self):
        width = max(map(len, self.calib)) + 1
        sig, pad = 6, 4*' '

        # number of points to left of decimal. '+3' factors in minus signs
        with np.errstate(divide='ignore', invalid='ignore'):
            # locally silence RuntimeWarnings about dividing by zero or infinity
            values = list(self.calib.values())
            l = int(np.fix(np.nanmax(np.log10(np.absolute(values)))) + 3)

        heading = 'ModelCalibration'
        string = [heading]

        for k, v in self.calib.items():
            row = '= '.join([pad+k.ljust(width), f"{v:{l+sig}.{sig}f}"])
            string.append(row)

        return '\n'.join(string)



def solve_calibration(calib):
    """
    given a dictionary of the form
        'param': float | int | string | AST,
    solve for each of the parameter values

    Parameters
    ----------
    calib : dict
        dictionary of numbers or variable definitions to evaluate

    Returns
    -------
    calib : dict
        dictionary of 'var_name': number
    """
    calib_ = {}
    for k, v in calib.items():
        if isinstance(v, (float, int)):
            calib_[k] = v
        elif isinstance(v, str):
            calib_[k] = parse_string(v)
        elif isinstance(v, ast.Num):
            calib_[k] = v.value
        elif isinstance(v, ast.AST):
            calib_[k] = v
        else:
            raise TypeError(
                f"{type(v)}. can only calibrate using numbers, strings, or ASTs"
            ) from None

    # create a dictionary of `var_name: vars it depends on`
    rhs_vars = {}
    for k, v in calib_.items():
        if isinstance(v, ast.AST):
            rhs_vars[k] = ParamLocator(v).locate()
        else:
            rhs_vars[k] = []

    # list the variables in the order they need to be solved
    precedence = order_precedence(rhs_vars)

    # fill in all the ASTs and return
    for var in precedence:
        calib_[var] = evaluate_parameter(calib_[var], calib_)
    return calib_



class ParamLocator(base.ABCVisitor):

    def __init__(self, tree: AST):
        super().__init__(tree)

        self.params = set()

    def locate(self):
        self.visit(self.tree)
        return self.params

    def visit_BinaryOp(self, node):
        self.visit(node.left)
        self.visit(node.right)

    def visit_UnaryOp(self, node):
        self.visit(node.expr)

    def visit_Num(self, node):
        pass

    def visit_Function(self, node):
        self.visit(node.expr)

    def visit_Var(self, node):
        var_name = node.value
        self.params.add(var_name)



def evaluate_parameter(
    expr: Union[float, int, AST],
    scope: dict
):
    """
    just the usual BaseEvaluator, except a type check for numbers is made
    prior to evaluation

    Parameters
    ----------
    expr : float | int | AST
        the expression to evaluate
    scope : dict
        a dictionary of parameter names and values

    Returns
    -------
    float | int
    """
    if isinstance(expr, (float, int)):
        return expr

    return base.BaseEvaluator(expr, scope)



def order_precedence(rhs_vars):
    """
    list the variables in order of solving

    Parameters
    ----------
    rhs_vars : dict
        a dictionary of the form 'var_name': (rhs_variables)

    Returns
    -------
    precedence : list
        list of keys of `rhs_vars` in the order of solving
    """
    # initialize dictionary of {var_name: order_of_solving}
    precedence = dict.fromkeys(rhs_vars.keys(), 0)
    for k, v in rhs_vars.items():
        if not v:
            precedence[k] = 1

    # number of params defined in terms of functions as other params
    n_funcs = len(precedence) - sum(precedence.values())
    total_iters = n_funcs * (n_funcs + 1) / 2

    while (n_funcs > 0) and (total_iters > 0):
        for k, rhs in rhs_vars.items():
            if precedence[k] == 0:

                # if all vars on the rhs have precedence, we know current var's
                if all([precedence[var] > 0 for var in rhs]):
                    precedence[k] = sum([precedence[var] for var in rhs])
                    n_funcs -= 1

        total_iters -= 1

    if any([p == 0 for p in precedence.values()]):
        raise CalibrationError(n_funcs)

    return sorted(precedence, key=precedence.get)
