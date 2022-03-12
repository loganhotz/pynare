"""
second-order approximation of the policy function
"""

from __future__ import annotations

from scipy import linalg

import numpy as np

import pynare.utils.numpy as unp
from pynare.core.solve.base import LinearSolution
from pynare.io.summary import SecondOrderSummary



class SecondOrderSolution(LinearSolution):

    order = 2
    __arrays__ = ('ghxx', 'ghxu', 'ghuu', 'del2')

    def __init__(self, model: Model):
        super().__init__(model)

        self.lower_solution = self.model.solve(order=1)

    def solve(self, *args, **kwargs):

        solver = SecondOrderSolver(self.model, self.lower_solution)

        ghxx, ghxu, ghuu, del2 = solver.solve()
        self.ghxx, self.ghxu, self.ghuu, self.del2 = ghxx / 2, ghxu, ghuu / 2, del2

        return self

    def summary(self, **kwargs):
        """
        returns a summary of the second-order solution to a model that displays the
        policy functions

        Parameters
        ----------
        sig : int = 6
            the number of digits to display when printing the policy matrices
        pad : int = 4
            not used in this summary

        Returns
        -------
        SecondOrderSummary
        """
        return SecondOrderSummary(self, **kwargs)



class SecondOrderSolver(object):

    def __init__(self, model: Model, lower_sol: FirstOrderSolution):
        self.model = model
        self.lower_solution = lower_sol

    def _setup_indices(self):
        """
        pre-allocate the different sets of indices used to reorder arrays before
        solving
        """

        indexes = self.model.indexes
        ns, nt = indexes.n_static, indexes.n_state

        # decision-rule indices of state vars
        self.tloc_arr = np.arange(ns, ns+nt, dtype=int)
        try:
            self.tloc = self.tloc_arr.item()
        except ValueError:
            self.tloc = self.tloc_arr

        # re-arrange lead-lag incidence to DR order
        dr_llx = indexes.dr.llx
        self.dr_back = dr_llx[0, :]
        self.dr_cont = dr_llx[1, :]
        self.dr_fore = dr_llx[2, :]

        # reordering blocks in the hessian from (declaration order, declaration order)
        #    to (dr order, dr order)
        edr = dr_llx[~np.isnan(dr_llx)]
        dr = np.concatenate((edr, len(edr)+np.arange(indexes.n_exog))).astype(int)

        # to do this hessian reordering, create index array where row index is the
        #    i-th block of hessian, column index is the j-th column of chosen block (both
        #    in declaration space), and value is target column idx in decision space
        self.dr_sq = np.arange(len(dr)**2, dtype=int).reshape(len(dr), len(dr))
        self.dr_dr = self.dr_sq[np.ix_(dr, dr)].flatten()


    def _setup_system(self, *args, **kwargs):
        """
        prepare the A, B, C, and D arrays used to solve the second-order
        linearized systems
        """
        # number of static and state variables
        indexes = self.model.indexes
        ns, nt = indexes.n_static, indexes.n_state

        # A, B, C, D, E arrays rely on first-order solution arrays
        gy = self.lower_solution.gy

        # reorder, get time t and t+1 indices
        jacobian = self.model.jacobian.array
        hess_dr = self.model.hessian.array[:, self.dr_dr]

        cont_jac = jacobian[:, _nonnan_int(self.dr_cont)]
        fore_jac = jacobian[:, _nonnan_int(self.dr_fore)]

        # construct the [dfdy' * dgdx + dfdx' | dfdy] array
        A = np.zeros((indexes.n_endog, indexes.n_endog), dtype=float)
        A[:, ~np.isnan(self.dr_cont)] = cont_jac
        tm1_update = unp.matmul_scalar(fore_jac, gy[~np.isnan(self.dr_fore)])
        A[:, self.tloc] = A[:, self.tloc] + tm1_update

        B = np.identity(nt**2, dtype=float)

        # construct the [0 | dfdy'] array
        C = np.zeros((indexes.n_endog, indexes.n_endog), dtype=float)
        C[:, -len(_nonnan_int(self.dr_fore)):] = fore_jac

        gy_state = gy[self.tloc]
        D = np.transpose(np.kron(gy_state, gy_state))

        return A, B, C, D


    def _hess_coefs_dx(self, lower_sol):
        """
        construct the coefficient matrices for the various blocks of the hessian
        matrix on the RHS of the sylvester equation. the coefficients are arranged
        to hit the columns of the hessian in the order
            [predetermined, time-t endo, time-(t+1) endo, exogenous]
        """

        indexes = self.model.indexes
        nt = indexes.n_state

        gy = lower_sol.gy

        state_eye = np.identity(nt, dtype=float)
        cont_gy = gy[~np.isnan(self.dr_cont)]
        fore_gy = unp.matmul_scalar(gy[~np.isnan(self.dr_fore)], gy[self.tloc])
        exo_block = np.zeros((indexes.n_exog, nt), dtype=float)

        E = unp.concatenate_maybe_1d((state_eye, cont_gy, fore_gy, exo_block))
        return E


    def _hess_coefs_du(self, lower_sol):
        """
        construct the coefficient matrices for solving for ghxu and ghuu. order
        is the same as that in `_hess_coefs_dx`
        """

        indexes = self.model.indexes
        nt, ne = indexes.n_state, indexes.n_exog

        gy = lower_sol.gy
        gu = lower_sol.gu

        state_zero = np.zeros((nt, ne), dtype=float)
        cont_gu = gu[~np.isnan(self.dr_cont)]

        try:
            fore_gy = unp.matmul_scalar(gy[~np.isnan(self.dr_fore)], gu[self.tloc])
        except ValueError:
            # one state var, > 1 stoch vars
            fore_gy = np.outer(gy[~np.isnan(self.dr_fore)], gu[self.tloc])

        exo_eye = np.identity(ne, dtype=float)

        E = unp.concatenate_maybe_1d((state_zero, cont_gu, fore_gy, exo_eye))
        return E


    def solve(self):

        self._setup_indices()
        A, B, C, D = self._setup_system()

        lower_sol = self.lower_solution
        zx = self._hess_coefs_dx(lower_sol)
        zu = self._hess_coefs_du(lower_sol)

        hess_dr = self.model.hessian.array[:, self.dr_dr]
        ghxx = solve_ghxx(A, B, C, D, zx, hess_dr)
        ghxu = solve_ghxu(lower_sol, self.tloc, A, C, ghxx, zx, zu, hess_dr)
        ghuu = solve_ghuu(lower_sol, self.tloc, A, C, ghxx, zu, hess_dr)

        indexes = self.model.indexes
        del2 = solve_del2(
            lower_sol,
            self.dr_cont, self.dr_fore, self.dr_sq,
            indexes.n_static, len(indexes.pure_forward_timed[1]),
            ghuu, self.model.sigma,
            self.model.jacobian.array, self.model.hessian.array
        )

        return ghxx, ghxu, ghuu, del2



def solve_ghxx(
    A: np.ndarray,
    B: np.ndarray,
    C: np.ndarray,
    D: np.ndarray,
    zx: np.ndarray,
    hess: np.ndarray
):
    """
    solve for the d^2G/dxdx term in the second-order approximation.

    Parameters
    ----------
    A, B, C, D : np.ndarray
        arrays that form left-hand side of the generalized sylvester equation
    zx : np.ndarray
        the coefficients that correspond to the state variables that interact
        with the hessian matrix entries
    hess : np.ndarray
        the dynamic hession of the model

    Returns
    -------
    ghxu : np.ndarray
    """
    E = -unp.matrix_kronecker_product(hess, zx)

    ghxx = unp.generalized_sylvester(A, B, C, D, E)
    return ghxx



def solve_ghxu(
    lower_sol: FirstOrderSolution,
    tloc: Union[np.ndarray, int],
    A: np.ndarray,
    C: np.ndarray,
    ghxx: np.ndarray,
    zx: np.ndarray,
    zu: np.ndarray,
    hess: np.ndarray
):
    """
    solve for the d^2G/dxdu term in the second-order approximation.

    Parameters
    ----------
    lower_sol : FirstOrderSolution
        the first-order solution to the model that contains the `gy` and `gu`
        arrays that define that linearized approximation
    tloc : np.ndarray | int
        the location (in DR order) of the state variables
    A : np.ndarray
        the (n_endog, n_endog) array `A` in the generalized sylvester equation
        that yields d^2G/dx^2
    C : np.ndarray
        the (n_endog, n_endog) array `C` in the generalized sylvester equation
        that yields d^2G/dx^2
    ghxx : np.ndarray
        the d^2G/dx^2 matrix term in the second-order approximation
    zx : np.ndarray
        the coefficients that correspond to the state variables that interact
        with the hessian matrix entries
    zu : np.ndarray
        the coefficients that correspond to the exogenous variables that interact
        with the hessian matrix entries
    hess : np.ndarray
        the dynamic hession of the model

    Returns
    -------
    ghxu : np.ndarray
    """
    # if theres only one state variable, zx will be a vector, and if there is
    #    only a single shock, zu will be too. for these to conform with one another
    #    in the kronecker product, ensure 2d
    zx, zu = unp.ensure_2darray((zx, zu))
    E = -unp.matrix_kronecker_product(hess, zx, zu).squeeze()

    # `matrix_kronecker_product` throws ValueErrors when scalars are passed or matrix
    #    dimensions depend on row/column orientation. just cast to 2dims for now
    gy, gu = lower_sol.gy, lower_sol.gu
    ghxx_arr, gy_arr, gu_arr = unp.ensure_2darray((ghxx, gy[tloc], gu[tloc]))

    if gu_arr.shape[1] == 1:
        # row-orient to conform with the following matrix product
        gu_arr = np.transpose(gu_arr)

    cross_prod = unp.matrix_kronecker_product(ghxx_arr, gy_arr, gu_arr).squeeze()

    rhs = E - unp.matmul_scalar(C, cross_prod)
    return linalg.solve(A, rhs)



def solve_ghuu(
    lower_sol: FirstOrderSolution,
    tloc: Union[np.ndarray, int],
    A: np.ndarray,
    C: np.ndarray,
    ghxx: np.ndarray,
    zu: np.ndarray,
    hess: np.ndarray
):
    """
    solve for the d^2G/du^2 term in the second-order approximation.

    Parameters
    ----------
    lower_sol : FirstOrderSolution
        the first-order solution to the model that contains the `gy` and `gu`
        arrays that define that linearized approximation
    tloc : np.ndarray | int
        the location (in DR order) of the state variables
    A : np.ndarray
        the (n_endog, n_endog) array `A` in the generalized sylvester equation
        that yields d^2G/dx^2
    C : np.ndarray
        the (n_endog, n_endog) array `C` in the generalized sylvester equation
        that yields d^2G/dx^2
    ghxx : np.ndarray
        the d^2G/dx^2 matrix term in the second-order approximation
    zu : np.ndarray
        the coefficients that correspond to the exogenous variables that interact
        with the hessian matrix entries
    hess : np.ndarray
        the dynamic hession of the model

    Returns
    -------
    ghuu : np.ndarray
    """
    E = -unp.matrix_kronecker_product(hess, zu)

    ghxx_tp1 = np.matmul(C, ghxx)
    state_du = lower_sol.gu[tloc]

    # if theres only one state variable, ghxx_tp1 will be a vector (if there is
    #    additionally only a single shock, state_du will be a scalar). for these
    #    to conform with one another in the kronecker product, ensure 2d
    ghxx_tp1, state_du = unp.ensure_2darray((ghxx_tp1, state_du))

    if state_du.shape[1] == 1:
        # row-orient to conform with the following matrix product
        state_du = np.transpose(state_du)

    ghxx_tp1_evaluated = unp.matrix_kronecker_product(ghxx_tp1, state_du).squeeze()

    rhs = E - ghxx_tp1_evaluated
    return linalg.solve(A, rhs)



def solve_del2(
    lower_sol: FirstOrderSolution,
    cont: np.ndarray,
    fore: np.ndarray,
    idx_sq: np.ndarray,
    ns: int,
    npf: int,
    ghuu: np.ndarray,
    sigma: np.ndarray,
    jac: np.ndarray,
    hess: np.ndarray
):
    """
    solve for the \Delta^2 term in the second-order approximation.

    Parameters
    ----------
    lower_sol : FirstOrderSolution
        the first-order solution to the model that contains the `gy` and `gu`
        arrays that define that linearized approximation
    cont : np.ndarray
        the time-t row of the lag-lead incidence array, in DR order
    fore : np.ndarray
        the time-(t+1) row of the lag-lead incidence array, in DR order
    idx_sq : np.ndarray
        the (n_endog+n_exog, n_endog+n_exog) square array of indices corresponding
        to the columns of the hessian matrix
    ns : int
        the number of static variables in the model
    npf : int
        the number of purely forward variables in the model
    ghuu : int
        the `ghuu` array in the second-order linear approximation
    sigma : int
        the square matrix of stochastic shock variances
    jac : np.ndarray
        the dynamic jacobian of the model
    hess : np.ndarray
        the dynamic hession of the model

    Returns
    -------
    del2 : np.ndarray
    """

    gy, gu = lower_sol.gy, lower_sol.gu

    fore_nan = np.isnan(fore)
    fore_idx = _nonnan_int(fore)
    fore_sq = idx_sq[np.ix_(fore_idx, fore_idx)].flatten()

    B1 = unp.matrix_kronecker_product(hess[:, fore_sq], gu[~fore_nan])

    rhs = np.matmul(jac[:, fore_idx], ghuu[~fore_nan]) + B1
    rhs = -unp.matmul_scalar(rhs, np.transpose(sigma).flatten())


    n_fore = np.count_nonzero(~fore_nan)
    n_endog = jac.shape[0]
    endo_eye = np.identity(n_endog, dtype=float)

    static_zero = np.zeros((n_fore, ns), dtype=float)
    pfore_zero = np.zeros((n_fore, npf), dtype=float)
    gy_fore = gy[~fore_nan]

    d_fore = unp.concatenate_maybe_1d((static_zero, gy_fore, pfore_zero), axis=1)
    deriv_fore = endo_eye[~fore_nan] + d_fore

    lhs = np.zeros((n_endog, n_endog), dtype=float)
    lhs[:, ~np.isnan(cont)] = jac[:, _nonnan_int(cont)]
    lhs = lhs + np.matmul(jac[:, fore_idx], deriv_fore)

    return linalg.solve(lhs, rhs) / 2


def _nonnan_int(arr):
    """
    utility function for translating index arrays with nans (which forces them
    to be stored as floats) to ints
    """
    return arr[~np.isnan(arr)].astype(int)
