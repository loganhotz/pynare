"""
implementing the solution method for models with binding constraints, including
occasionally binding constraints
"""
from __future__ import annotations

from scipy import linalg

import numpy as np
import warnings

from pynare.parsing import parse_string, evaluate
from pynare.regimes import RegimeSwitchingModel

from pynare.core.solve.base import ModelSolution
from pynare.utils.dtypes import is_iterable_not_str

from pynare.errors import ModelVarError



class Constraint(object):
    """
    constraints on the endogenous variables of a model. currently can only interpret
    occbin-style constraints
    """

    def __init__(
        self,
        constraint: Sequence[str | Sequence[str]],
        endog: Sequence[str | EndogVar] = None,
        params: Mapping = {},
        exog: Sequence[str | ExogVar] = None
    ):
        # save the properties needed for evaluating Constraint when called
        try:
            self.endog = endog.names
        except AttributeError:
            self.endog = endog

        try:
            self.exog = exog.names
        except AttributeError:
            self.exog = exog

        self.params = params

        # interpret constraints
        if is_iterable_not_str(constraint):

            if is_iterable_not_str(constraint[0]):
                # assume `constraints` is a sequence of constraints pairs
                self.ref, self.alt, self.ref_ast, self.alt_ast = [], [], [], []

                for con in constraint:
                    if len(con) != 2:
                        raise ValueError(
                            "each constraint condition must be an iterable of "
                            "length 2"
                        ) from None

                    self.ref.append(con[0])
                    self.alt.append(con[1])

                    self.ref_ast.append(parse_string(con[0]))
                    self.alt_ast.append(parse_string(con[1]))

            else:
                # assume `constraints` is a 2-tuple of reference and alternative
                #   switch conditions
                if len(constraint) != 2:
                    raise ValueError(
                        "if there is only one constraint, 'constraint' must be an "
                        "iterable of length 2"
                    ) from None

                # expressions that, when they evaluate to true, we switch regimes
                self.ref = constraint[0]
                self.alt = constraint[1]

                self.ref_ast = parse_string(constraint[0])
                self.alt_ast = parse_string(constraint[1])

        else:
            raise TypeError(
                f"{type(constraint)}. 'constraint' must be a sequence of str, or a "
                "sequence of sequence of str"
            ) from None

    def __len__(self):
        if is_iterable_not_str(self.ref):
            return len(self.ref)
        return 1

    def __str__(self):
        delim = '    '
        if len(self) > 1:
            pairs = [', '.join(pair) for pair in zip(self.ref, self.alt)]
            constraints = f"\n{delim}" + f"\n{delim}".join(pairs)
            return f"Constraint({constraints}\n)"

        else:
            return f"Constraint({self.ref}, {self.alt})"

    def __repr__(self):
        return f"Constraint(n={len(self)})"

    def __call__(self, paths: np.ndarray):
        # this is a very inefficient way of doing this, i think...
        #   combine simulated paths with the param values for a complete scope
        endog_paths = dict(zip(self.endog, np.hsplit(paths, paths.shape[1])))
        params = {**endog_paths, **self.params}

        if is_iterable_not_str(self.ref):
            constrs = []
            for ref, alt in zip(self.ref_ast, self.alt_ast):
                ref_bool = evaluate(ref, params)
                alt_bool = evaluate(alt, params)
                constrs.append([ref_bool, alt_bool])

            return np.block(constrs)

        else:

            ref = evaluate(self.ref_ast, params)
            alt = evaluate(self.alt_ast, params)

            return np.hstack((ref, alt))



class BindingSolution(ModelSolution):
    pass



class OccBinSolution(BindingSolution):

    def __init__(
        self,
        model: Model,
        constraint: Constraint | Sequence[str | Sequence[str]],
        exprs: dict = {},
        sigma: dict = {},
        exog: str | Sequence[str] = None,
        shocks: int | Sequence[float] = 100,
        init: np.ndarray = None
    ):
        if exprs:
            self.model = RegimeSwitchingModel(
                model,
                exprs=exprs,
                sigma=sigma
            )

        else:
            self.model = model

        self.constraint = Constraint(
            constraint,
            endog=model.endog,
            params=model.params,
            exog=model.stoch
        )

        # occbin solutions are conditional on the shocks to the model & the starting
        #   location of the endog vars, so we receive them upon initialization
        _, shocks, init, _ = _process_occbin_parameters(
            ref=self.model[0], # assume first model is reference regime
            exog=exog,
            shocks=shocks,
            init=init,
            periods=None
        )

        self.exog = exog
        self.shocks = shocks
        self.init = init

        # endog var paths under the piece-wise linear solution
        self.paths = None


    def solve(
        self,
        periods: int = None,
        max_iter: int = 20,
        restrained: bool = False
    ):
        md = self.model
        if not isinstance(md, RegimeSwitchingModel):
            raise NotImplementedError(
                "can only create occbin solutions with regime-switching models"
            ) from None

        if len(self.constraint) == 1:
            paths, shocks = solve_occbin_one_constraint(
                ref=md.models[0], alt=md.models[1],
                constraint=self.constraint,
                exog=self.exog,
                shocks=self.shocks,
                init=self.init,
                periods=periods,
                max_iter=max_iter
             )

        else:
            mds = md.models
            paths, shocks = solve_occbin_two_constraints(
                ref=mds[0], alt0=mds[1], alt1=mds[2], dbl=mds[3],
                constraint=self.constraint,
                exog=self.exog,
                shocks=self.shocks,
                init=self.init,
                periods=periods,
                max_iter=max_iter,
                restrained=restrained
            )

        # the provided 'periods' may have been longer than the shocks that initialized
        #   the solution, so we overwrite them
        self.paths, self.shocks = paths, shocks
        return self



def solve_occbin_one_constraint(
    ref: Model,
    alt: Model,
    constraint: Constraint,
    exog: str | Sequence[str] = None,
    shocks: int | tuple | np.ndarray = 100,
    init: np.ndarray = None,
    periods: int = None,
    max_iter: int = 20
):
    """
    implements the solution algorithm outlined in Guerrieri & Iacoviello (2014) when
    only one constraint is imposed on the model

    Parameters
    ----------
    ref : Model
        the model corresponding to what G&I term the 'reference' regime
    alt : Model
        the model corresponding to what G&I term the 'alternative' regime
    constraint : Constraint
        a Constraint instance that determines when to switch between the two regimes
    exog : str | Sequence[str] ( = None )
        the names of the exog vars that are subject to shocks. if 'None' (default),
        it is assumed the first declared exog var is the only one that is shocked
    shocks : int | tuple | ndarray ( = 100 )
        the shocks afflicting the exog vars provided in `exog`. if 'int' or 'tuple',
        an array of that shape is created with `numpy.random.normal`. otherwise, an
        array-like of numbers can be passed
    init : ndarray ( = None )
        the initial positions of the endog vars. if 'None' (default), they are all
        assumed to be equal to zero
    periods : int ( = None )
        the number of simulation periods. it must be large enough to ensure arrival
        back at the reference regime at the end of the simulation
    max_iter : int ( = 20 )
        the maximum number of attempts for the solution algorithm in each period

    Returns
    -------
    (paths, shocks) : 2-tuple of ndarrays
        the paths of the endogenous variables implied by the piecewise solution, and
        the vector/array of shocks the model was subject to
    """

    if len(constraint) != 1:
        raise ValueError(f"n constraints = {len(constraints)}")

    # throws an error if endog, stoch, or params are misaligned between models
    _ensure_variables_match(ref, alt)

    # ensure 'shocks' is a vector or array, fill out 'init' if not given, provide
    #   a default value for 'periods', and check solution order of 'ref'
    exog_idx, shocks, init, periods = _process_occbin_parameters(
        ref=ref,
        exog=exog,
        shocks=shocks,
        init=init,
        periods=periods
    )

    # get the first-order decision rules in declaration order
    idx = ref.indexes

    gy_dc = np.zeros((idx.n_endog, idx.n_endog), dtype=float)
    gy_dc[:, idx.state] = ref.first_order_solution.arrays.gy[idx.dc_order]
    gu_dc = ref.first_order_solution.arrays.gu[idx.dc_order]

    # partition the jacobian matrix of each regime
    C, B, A, E = _occbin_partitioned_jacobian(ref)
    Cstar, Bstar, Astar, Estar = _occbin_partitioned_jacobian(alt)

    # correction for alt regime due to linearizing around the reference one
    Dstar = alt.dynamic_repr(_translate_ss(ref.ss.values, alt))

    # pre-allocate the piecewise linear paths and the regime indicators. save
    #   initial endogenous vector, whose name will be overwritten
    paths = np.zeros((periods, len(ref.endog)), dtype=float)
    hypo = np.zeros(periods+1, dtype=bool)
    provided_init = init.copy()

    # identify existing regime for each period
    for idx, shk in enumerate(shocks):

        regime_change = True
        current_iter = 0

        while regime_change and (current_iter < max_iter):

            current_iter += 1

            # analyze when each regime begins based on the current guess & compute
            #   the hypothesized piecewise linear solution based on that guess
            regime_starts, latest_regime = _locate_regime_changes(hypo)
            guess = _simulate_hypothesized_paths(
                gy=gy_dc, gu=gu_dc,
                A=A, B=B, C=C, E=E,
                Astar=Astar, Bstar=Bstar, Cstar=Cstar, Estar=Estar, Dstar=Dstar,
                regime_starts=regime_starts, regime=hypo, periods=periods,
                exog_idx=exog_idx, init=init, shock=shk
            )

            # create a (period+1, 2) array of booleans with indicators of a binding
            #   constraint (col 1) or not being constrained (col 2)
            constr = constraint(guess)

            # check for mismatches between being in the alternative regime and whether
            #   or not the constraint is binding for those same periods
            if ~np.all(hypo[constr[:, 0]]) or np.any(constr[hypo, 1]):
                regime_change = True
            else:
                regime_change = False

            alt_const = np.logical_or(hypo, constr[:, 0])
            alt_relaxed = np.logical_and(hypo, constr[:, 1])
            hypo = np.greater(alt_const, alt_relaxed)

        init = guess[0]
        paths[idx] = init

        # reset hypothesis - enforces assumption that we expect no further shocks
        hypo = np.append(hypo[1:], False)

        # if 'regime_change' is True, the prevailing regime in the final period is the
        #   althernative one
        if regime_change:
            warnings.warn("algorithm did not converge. increase 'max_iter'")

    # use remainder of the final guess array to fill out rest of solution
    if len(shocks) < periods:
        diff = periods - len(shocks)
        paths[-diff:] = guess[1:diff+1]

        try:
            n_shock_periods, n_shocks = shocks.shape
            zeros = np.zeros((periods - n_shock_periods, n_shocks), dtype=float)
        except ValueError:
            zeros = np.zeros(periods - len(shocks), dtype=float)

        shocks = np.append(shocks, zeros, axis=0)

    return paths, shocks



def solve_occbin_two_constraints(
    ref: Model,
    alt0: Model,
    alt1: Model,
    dbl: Model,
    constraint: Constraint,
    exog: str | Sequence[str] = None,
    shocks: int | tuple | np.ndarray = 100,
    init: np.ndarray = None,
    periods: int = None,
    max_iter: int = 20,
    restrained: bool = False
):
    """
    implements the solution algorithm outlined in Guerrieri & Iacoviello (2014) when
    two constraints are imposed on the model

    Parameters
    ----------
    ref : Model
        the model corresponding to what G&I term the 'reference' regime
    alt0 : Model
        the model corresponding for the alternative regime when constraint one binds
    alt1 : Model
        the model corresponding for the alternative regime when constraint two binds
    dbl : Model
        the model used when both constraints bind
    constraint : Constraint
        a Constraint instance that determines when to switch between the two regimes
    exog : str | Sequence[str] ( = None )
        the names of the exog vars that are subject to shocks. if 'None' (default),
        it is assumed the first declared exog var is the only one that is shocked
    shocks : int | tuple | ndarray ( = 100 )
        the shocks afflicting the exog vars provided in `exog`. if 'int' or 'tuple',
        an array of that shape is created with `numpy.random.normal`. otherwise, an
        array-like of numbers can be passed
    init : ndarray ( = None )
        the initial positions of the endog vars. if 'None' (default), they are all
        assumed to be equal to zero
    periods : int ( = None )
        the number of simulation periods. it must be large enough to ensure arrival
        back at the reference regime at the end of the simulation
    max_iter : int ( = 20 )
        the maximum number of attempts for the solution algorithm in each period
    restrained: bool ( = False )
        indicator for following Gauss-Jacobi pattern of updating only one constraint
        at a time

    Returns
    -------
    (paths, shocks) : 2-tuple of ndarrays
        the paths of the endogenous variables implied by the piecewise solution, and
        the vector/array of shocks the model was subject to
    """

    if len(constraint) != 2:
        raise ValueError(f"n constraints = {len(constraint)}")

    # throws an error if endog, stoch, or params are misaligned between models
    _ensure_variables_match(ref, alt0)
    _ensure_variables_match(ref, alt1)
    _ensure_variables_match(ref, dbl)

    # ensure 'shocks' is a vector or array, fill out 'init' if not given, provide
    #   a default value for 'periods', and check solution order of 'ref'
    exog_idx, shocks, init, periods = _process_occbin_parameters(
        ref=ref,
        exog=exog,
        shocks=shocks,
        init=init,
        periods=periods
    )

    # get the first-order decision rules in declaration order
    ridx = ref.indexes

    gy_dc = np.zeros((ridx.n_endog, ridx.n_endog), dtype=float)
    gy_dc[:, ridx.state] = ref.first_order_solution.arrays.gy[ridx.dc_order]
    gu_dc = ref.first_order_solution.arrays.gu[ridx.dc_order]

    # partition the jacobian matrix of each regime
    C, B, A, E = _occbin_partitioned_jacobian(ref)
    C_alt0, B_alt0, A_alt0, E_alt0 = _occbin_partitioned_jacobian(alt0)
    C_alt1, B_alt1, A_alt1, E_alt1 = _occbin_partitioned_jacobian(alt1)
    C_dbl, B_dbl, A_dbl, E_dbl = _occbin_partitioned_jacobian(dbl)

    # correction for alt regimes due to linearizing around the reference one
    D_alt0 = alt0.dynamic_repr(_translate_ss(ref.ss.values, alt0))
    D_alt1 = alt1.dynamic_repr(_translate_ss(ref.ss.values, alt1))
    D_dbl = dbl.dynamic_repr(_translate_ss(ref.ss.values, dbl))

    # pre-allocate the piecewise linear paths and the regime indicators. save
    #   initial endogenous vector, whose name will be overwritten
    paths = np.zeros((periods, len(ref.endog)), dtype=float)
    hypo = np.zeros((periods+1, 2), dtype=bool)
    provided_init = init.copy()

    # identify existing regime for each period
    for idx, shk in enumerate(shocks):

        regime_change = True
        current_iter = 0

        while regime_change and (current_iter < max_iter):

            current_iter += 1

            # analyze when each regime begins based on the current guess & compute
            #   the hypothesized piecewise linear solution based on that guess
            regime_starts0, latest_regime0 = _locate_regime_changes(hypo[:, 0])
            regime_starts1, latest_regime1 = _locate_regime_changes(hypo[:, 1])

            guess = _simulate_hypothesized_paths_two_constraints(
                gy=gy_dc, gu=gu_dc,
                A=A, B=B, C=C, E=E,
                A_alt0=A_alt0, B_alt0=B_alt0, C_alt0=C_alt0,
                E_alt0=E_alt0, D_alt0=D_alt0,
                A_alt1=A_alt1, B_alt1=B_alt1, C_alt1=C_alt1,
                E_alt1=E_alt1, D_alt1=D_alt1,
                A_dbl=A_dbl, B_dbl=B_dbl, C_dbl=C_dbl,
                E_dbl=E_dbl, D_dbl=D_dbl,
                regime_starts0=regime_starts0, regime_starts1=regime_starts1,
                regime0=hypo[:, 0], regime1=hypo[:, 1],
                periods=periods, exog_idx=exog_idx, init=init, shock=shk
            )

            # create a (2*(period+1), 2) array of booleans with indicators of a binding
            #   constraint (col 1) or not being constrained (col 2). the top half of
            #   each column relates to the first constraint; the bottom, the second
            constr = constraint(guess)
            long_hypo = hypo.ravel(order='F')

            # check for mismatches between being in the alternative regime and whether
            #   or not the constraint is binding for those same periods
            if ~np.all(long_hypo[constr[:, 0]]) or np.any(constr[long_hypo, 1]):
                regime_change = True
            else:
                regime_change = False

            # optionally update one constraint at a time
            if restrained:
                rest = np.zeros_like(hypo)
                constr0, constr1 = np.split(constr, 2, axis=0)

                rest[np.logical_and(hypo[:, 0], constr0[:, 1]), 0] = True
                rest[np.logical_and(hypo[:, 1], constr1[:, 1]), 1] = True

                joint_hypo = np.logical_or(long_hypo, constr[:, 0])
                hypo = np.greater(joint_hypo.reshape((-1, 2), order='F'), rest)

            else:
                alt_const = np.logical_or(long_hypo, constr[:, 0])
                alt_relaxed = np.logical_and(long_hypo, constr[:, 1])
                hypo = np.greater(alt_const, alt_relaxed).reshape((-1, 2), order='F')

        init = guess[0]
        paths[idx] = init

        # reset hypothesis - enforces assumption that we expect no further shocks
        hypo = np.append(hypo[1:], [[False, False]], axis=0)

        # if 'regime_change' is True, the prevailing regime in the final period is the
        #   althernative one
        if regime_change:
            warnings.warn("algorithm did not converge. increase 'max_iter'")

    # use remainder of the final guess array to fill out rest of solution
    if len(shocks) < periods:
        diff = periods - len(shocks)
        paths[-diff:] = guess[1:diff+1]

        try:
            n_shock_periods, n_shocks = shocks.shape
            zeros = np.zeros((periods - n_shock_periods, n_shocks), dtype=float)
        except ValueError:
            zeros = np.zeros(periods - len(shocks), dtype=float)

        shocks = np.append(shocks, zeros, axis=0)

    return paths, shocks



def _simulate_hypothesized_paths(
    gy: np.ndarray,
    gu: np.ndarray,
    A: np.ndarray,
    B: np.ndarray,
    C: np.ndarray,
    E: np.ndarray,
    Astar: np.ndarray,
    Bstar: np.ndarray,
    Cstar: np.ndarray,
    Estar: np.ndarray,
    Dstar: np.ndarray,
    regime_starts: np.ndarray[int],
    regime: np.ndarray[bool],
    periods: int,
    exog_idx: int | Sequence[int],
    init: np.ndarray,
    shock: float
):

    try:
        n_endog, n_exog = gu.shape
    except ValueError:
        n_endog, n_exog = len(gu), 1

    # final period with a binding constraint
    last_bound = regime_starts[-1] - 1

    # if we enter into the alternative regime, begin the solution algorithm
    if last_bound > -1:
        P = np.zeros((n_endog, n_endog, last_bound+1), dtype=float)
        D = np.zeros((n_endog, last_bound+1), dtype=float)

        # first part in step two of solution algorithm
        last_inv_mat = linalg.inv(np.matmul(Astar, gy) + Bstar)
        P[:, :, -1] = -np.matmul(last_inv_mat, Cstar)
        D[:, -1] = -np.matmul(last_inv_mat, Dstar)

        for i in range(last_bound-1, -1, -1):
            # final part in step two
            if regime[i]:
                inv_mat = linalg.inv(np.matmul(Astar, P[:,:,i+1]) + Bstar)
                P[:, :, i] = -np.matmul(inv_mat, Cstar)
                D[:, i] = -np.matmul(inv_mat, np.matmul(Astar, D[:,i+1]) + Dstar)

            else:
                inv_mat = linalg.inv(np.matmul(A, P[:,:,i+1]) + B)
                P[:, :, i] = -np.matmul(inv_mat, C)
                D[:, i] = -np.matmul(inv_mat, np.matmul(A, D[:,i+1]))

        # step five in solution algo
        if last_bound > 0:
            E_regime = Estar if regime[0] else E
            E_init = -np.matmul(inv_mat, E_regime)

        else:
            E_init = -np.matmul(last_inv_mat, Estar)

    # hypothesized starting point
    history = np.zeros((periods+2, n_endog), dtype=float)
    history[0] = init

    # predetermined innovations
    innovs = np.zeros(n_exog, dtype=float)
    if isinstance(exog_idx, int):
        innovs[exog_idx] = shock
    else:
        for i, ex in enumerate(exog_idx):
            innovs[ex] = shock[i]

    try:
        innovs = innovs.item()
    except ValueError:
        pass

    # shocks to exogenous variables
    if last_bound > -1:
        Q = np.squeeze(np.dot(E_init, innovs))
        history[1] = np.dot(P[:, :, 0], history[0]) + D[:, 0] + Q
    else:
        history[1] = np.dot(gy, history[0]) + np.dot(gu, innovs)

    # assumed that no further shocks are realized
    for i in range(1, periods+1):
        if i <= last_bound:
            history[i+1] = np.dot(P[:, :, i], history[i]) + D[:, i]
        else:
            history[i+1] = np.dot(gy, history[i])

    return history[1:]



def _simulate_hypothesized_paths_two_constraints(
    gy: np.ndarray,
    gu: np.ndarray,
    A: np.ndarray, B: np.ndarray, C: np.ndarray, E: np.ndarray,
    A_alt0: np.ndarray, B_alt0: np.ndarray, C_alt0: np.ndarray,
    E_alt0: np.ndarray, D_alt0: np.ndarray,
    A_alt1: np.ndarray, B_alt1: np.ndarray, C_alt1: np.ndarray,
    E_alt1: np.ndarray, D_alt1: np.ndarray,
    A_dbl: np.ndarray, B_dbl: np.ndarray, C_dbl: np.ndarray,
    E_dbl: np.ndarray, D_dbl: np.ndarray,
    regime_starts0: np.ndarray[int], regime_starts1: np.ndarray[int],
    regime0: np.ndarray[bool], regime1: np.ndarray[bool],
    periods: int,
    exog_idx: int | Sequence[int],
    init: np.ndarray,
    shock: float
):

    try:
        n_endog, n_exog = gu.shape
    except ValueError:
        n_endog, n_exog = len(gu), 1

    # final period with a binding constraint
    last_bound = np.amax((regime_starts0[-1], regime_starts1[-1])) - 1

    # if we enter into the alternative regime, begin the solution algorithm
    if last_bound > -1:
        P = np.zeros((n_endog, n_endog, last_bound+1), dtype=float)
        D = np.zeros((n_endog, last_bound+1), dtype=float)

        # first part in step two of solution algorithm
        if regime0[last_bound] and regime1[last_bound]:
            a, b, c, d = A_dbl, B_dbl, C_dbl, D_dbl
        elif regime0[last_bound]:
            a, b, c, d = A_alt0, B_alt0, C_alt0, D_alt0
        else:
            a, b, c, d = A_alt1, B_alt1, C_alt1, D_alt1

        last_inv_mat = linalg.inv(np.matmul(a, gy) + b)
        P[:, :, -1] = -np.matmul(last_inv_mat, c)
        D[:, -1] = -np.matmul(last_inv_mat, d)

        # final part in step two
        for i in range(last_bound-1, -1, -1):

            if regime0[i] and regime1[i]:
                a, b, c, d = A_dbl, B_dbl, C_dbl, D_dbl
            elif regime0[i]:
                a, b, c, d = A_alt0, B_alt0, C_alt0, D_alt0
            elif regime1[i]:
                a, b, c, d = A_alt1, B_alt1, C_alt1, D_alt1
            else:
                a, b, c, d = A, B, C, np.zeros(n_endog, dtype=float)

            inv_mat = linalg.inv(np.matmul(a, P[:, :, i+1]) + b)
            P[:, :, i] = -np.matmul(inv_mat, c)
            D[:, i] = -np.matmul(inv_mat, np.matmul(a, D[:, i+1]) + d)

        # step five in solution algo
        if last_bound > 0:
            if regime0[0] and regime1[0]:
                E_init = -np.matmul(inv_mat, E_dbl)
            elif regime0[0]:
                E_init = -np.matmul(inv_mat, E_alt0)
            elif regime1[0]:
                E_init = -np.matmul(inv_mat, E_alt1)
            else:
                E_init = -np.matmul(inv_mat, E)

        else:
            if regime0[last_bound] and regime1[last_bound]:
                E_init = -np.matmul(last_inv_mat, E_dbl)
            elif regime0[last_bound]:
                E_init = -np.matmul(last_inv_mat, E_alt0)
            else:
                E_init = -np.matmul(last_inv_mat, E_alt1)

    # hypothesized starting point
    history = np.zeros((periods+2, n_endog), dtype=float)
    history[0] = init

    # predetermined innovations
    innovs = np.zeros(n_exog, dtype=float)
    if isinstance(exog_idx, int):
        innovs[exog_idx] = shock
    else:
        for i, ex in enumerate(exog_idx):
            innovs[ex] = shock[i]

    try:
        innovs = innovs.item()
    except ValueError:
        pass

    # shocks to exogenous variables
    if last_bound > -1:
        Q = np.squeeze(np.dot(E_init, innovs))
        history[1] = np.dot(P[:, :, 0], history[0]) + D[:, 0] + Q
    else:
        history[1] = np.dot(gy, history[0]) + np.dot(gu, innovs)

    # assumed that no further shocks are realized
    for i in range(1, periods+1):
        if i < last_bound+1:
            # print(i)
            history[i+1] = np.dot(P[:, :, i], history[i]) + D[:, i]
        else:
            history[i+1] = np.dot(gy, history[i])

    return history[1:]



def _process_occbin_parameters(
    ref: Model,
    exog: str | Sequence[str],
    shocks: int | tuple | np.ndarray,
    init: np.ndarray,
    periods: int = None,
):
    """
    the single and double-constraint processes for computing occbin solutions require
    the exact same checks to be run before solving. this utility function implements
    those checks
    """

    # assume the model is hit by random normal shocks if an int or tuple is passed
    if isinstance(shocks, (int, tuple)):
        shocks = np.random.normal(size=shocks)
    else:
        try:
            shocks = np.asarray(shocks, dtype=float)
        except:
            # not a big fan of blind exceptions...
            raise TypeError(
                f"{type(shocks)}. 'shocks' must be int, tuple, or array-like of numbers"
            ) from None

    # assume number of periods is equal to the length of 'shocks' if not provided
    periods = len(shocks) if not periods else periods

    # the first values of the endogenous variables over the time period
    if init is None:
        init = np.zeros(len(ref.endog), dtype=float)

    # by default, we shock the first stochastic variable
    if exog is None:
        exog_idx = 0
    else:
        exog_idx = ref.stoch.get_loc(exog)

    # make sure 'exog' and 'shocks' comport
    try:
        n, m = 1 if isinstance(exog_idx, int) else len(exog_idx), shocks.shape[1]
        if n != m:
            raise ValueError(
                f"'exog' implies {n} exog vars, but the size of 'shocks' implies {m}"
            ) from None

    except IndexError:
        if not isinstance(exog_idx, int):
            n = len(exog_idx)
            raise ValueError(
                f"'exog' implies {n} exog vars, but the size of 'shocks' implies 1"
            ) from None

    # the Guerrieri & Iacoviello paper is only for first-order solutions
    if ref.solution_order:
        if ref.solution_order != 1:
            msg = (
                f"{ref} was previously solved at order = {ref.solution_order}. "
                "occassionally binding solutions can only be solved for order = 1"
            )
            warnings.warn(msg)

    else:
        _ = ref.solve(order=1)

    return exog_idx, shocks, init, periods


def _occbin_partitioned_jacobian(model: Model):
    """
    the occbin algorithm partitions the jacobian matrix column-wise based on what time
    period the endogenous variable corresponding to that column appears in
    """
    J, idx = model.jacobian.array, model.indexes

    # partition the jacobian based on whether the column is a derivative with respect
    #   to time t-1, t, or t+1 variables
    J_tm1 = np.zeros((idx.n_endog, idx.n_endog), dtype=float)
    J_tm1[:, idx.backward] = J[:, idx.backward_timed[-1]]

    J_t = np.zeros((idx.n_endog, idx.n_endog), dtype=float)
    J_t[:, idx.cont] = J[:, idx.cont_timed[0]]

    J_tp1 = np.zeros((idx.n_endog, idx.n_endog), dtype=float)
    J_tp1[:, idx.forward] = J[:, idx.forward_timed[1]]

    # last partition is the jacobian w.r.t the exogenous variables
    J_exog = J[:, idx.exogenous_jacobian]

    return J_tm1, J_t, J_tp1, J_exog



def _locate_regime_changes(vec: np.ndarray):
    """
    locate the periods in which the model switches between regimes

    Parameters
    ----------
    vec : np.ndarray
        a vector of bools whose entries are interpreted as different periods of a
        model's solution path. a value of False means that period is in the
        'reference' regime, to use Guerrieri & Iacoviello's terminology

    Returns
    -------
    (idx, reg) : 2-tuple of vectors
        `idx` records the index of `vec` at which the regime switches
        `reg` records which regime is prevailing in the periods of `reg`
    """
    latest_regime = vec[:1]
    regime_starts = np.array([0], dtype=int)

    n_switches = 0
    for i in range(1, len(vec)-1):

        if vec[i] != latest_regime[n_switches]:
            n_switches += 1
            latest_regime = np.append(latest_regime, vec[i])
            regime_starts = np.append(regime_starts, i)

    # run the checks that are required for assumption (2) in section 2 of the paper
    if latest_regime[0] and (len(regime_starts) == 1):
        msg = (
            "increase 'nperiods': the model begins in the alternative regime and "
            "does not leave."
        )
        warnings.warn(msg, category=RuntimeWarning)

    elif latest_regime[-1]:
        # use elif so two warnings are not issued
        msg = "increase 'periods': the model ends in the alternative regime"
        warnings.warn(msg, category=RuntimeWarning)

    return regime_starts, latest_regime



def _ensure_variables_match(
    md: Model,
    other: Model
):
    """
    ensure the two models have the same endogenous and stochastic variables, and
    enforcing they are declared in the same order (a necessary consequence of how
    the `gy` and `gu` arrays are constructed). the params names of each need to be
    the same as well, although their order is immaterial
    """

    def _fmt(md1, md2):
        if md1.name and md2.name:
            return f"models '{md1.name}' and '{md2.name}'"
        return "both models"

    if not np.array_equal(md.endog.names, other.endog.names):
        raise ModelVarError(
            f"{_fmt(md, other)} must have the same endog vars, & in the same order"
        )

    if not np.array_equal(md.stoch.names, other.stoch.names):
        raise ModelVarError(
            f"{_fmt(md, other)} must have the same stoch vars, & in the same order"
        )

    # order doesn't matter for model parameters
    sorted_md = np.sort(np.array(list(md.params.keys())))
    sorted_other = np.sort(np.array(list(other.params.keys())))
    if not np.array_equal(sorted_md, sorted_other):
        raise ModelVarError(f"{_fmt(md, other)} must have the same params")



def _translate_ss(ref_ss, alt):
    """
    construct reference's steady state in terms of lead & lags of alternative model
    """
    idx = alt.indexes
    return np.take_along_axis(
        ref_ss,
        np.concatenate((idx.backward, idx.cont, idx.forward)),
        axis=None
    )
