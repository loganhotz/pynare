"""
first-order approximation of the policy function
"""

from __future__ import annotations

import numpy as np

from scipy import linalg

import pynare.utils.numpy as unp
from pynare.core.solve.base import LinearSolution
from pynare.errors import ModelIdentificationError
from pynare.io.summary import FirstOrderSummary



class FirstOrderSolution(LinearSolution):

    order = 1
    __arrays__ = ('gy', 'gu')

    def solve(self, *args, **kwargs):
        self.gy, self.gu = solve_first_order(self.model)
        return self

    def summary(self, **kwargs):
        """
        returns a summary of the solution that displays the first-order policy
        functions of the model.

        Parameters
        ----------
        sig : int = 6
            the number of digits to display when printing policy matrices
        pad : int = 4
            not used in this summary

        Returns
        -------
        FirstOrderSummary
        """
        return FirstOrderSummary(self, **kwargs)



def solve_first_order(
    M: Model
) -> Tuple[np.ndarray]:
    """
    Computes the first-order linear solution to a model. Notation follows
    Villemot (2011). A nonlinear model can be written as
                f(y_tp1^p, y_t, y_tm1^m, u_t) = 0,
    where
        y_t     - vector of endogenous variables
        y_t^p     - subset of y_t that appears with a lead
        y_t^m     - subset of y_t that appears with a lag
        u_t     - vector of exogenous shocks.
    If the model is expressed as above, the solution g solves the functional
    equation
                f(g(g(y_tm1, u_t), u_tp1), g(y_tm1, u_t), y_tm1, u_t) = 0.
    This function solves for the first-order solution, i.e.
                g(y_tm1, u_t) = ybar + gy*yhat_tm1 + gu*uhat_t
    where the 'hat' suffix denotes percent deviations from the model's steady
    state, ybar.

    Parameters
    ----------
    M : Model
        a pynare model

    Returns
    -------
    tuple of numpy ndarrays
        the tuple of vectors gy, gu described above
    """
    indexes = M.indexes
    state = M.statespace

    # retrieve state space matrices
    A, D, E = state.jacobian, state.lhs, state.rhs

    # solving for backward- and forward-looking dynamic variables and merging the arrays
    #   by construction the bottom n_mixed rows of gym match the top n_mixed of gyp
    gym, gyp = solve_nonstatic_endo(state, indexes)
    gyd = unp.array_shingle(gym, gyp, indexes.n_mixed)

    # solving for static variables, then stacking the static and dynamic arrays
    gys = solve_static_endo(A, indexes, gym, gyp)
    gy = np.vstack((gys, gyd))

    if indexes.n_exos > 0:
        fu = A[:, indexes.exogenous_jacobian]
    else:
        fu = J[:, indexes.exogenous_jacobian]
    gu = solve_exo(A, indexes, gyp, fu)

    # remove any 1-dimensional axes
    gy = gy.squeeze()
    gu = gu.squeeze()

    return gy, gu



def solve_nonstatic_endo(
    state: StateSpace,
    indexes: VariableIndexManager
):
    """
    Solving for the non-static endogenous variables of the model. This corresponds
    to Section 4.1 of Villemot (2011)

    Parameters
    ----------
    state : StateSpace
        the state-space representation of the model
    indexes : VariableIndexManager
        the index manager of the model

    Returns
    -------
    2-tuple of np.ndarray
        respectively, the state response to backward- and forward-looking dynamic
        endogenous variables
    """

    Z_11, Z_12, Z_21, Z_22 = state.zarrays
    T_11, S_11 = state.larrays.T_11, state.rarrays.S_11

    # solution for forward-looking variables
    gyp = - np.linalg.solve(Z_22, Z_21)

    # solution for backward-looking variables
    Z_11t = np.transpose(Z_11)
    gym = np.dot(
        np.dot(Z_11t, np.linalg.inv(T_11)),
        np.dot(S_11, np.linalg.inv(Z_11t))
    )

    return gym, gyp



def solve_static_endo(
    A: np.ndarray,
    indexes: VariableIndexManager,
    gym: np.ndarray,
    gyp: np.ndarray
):
    """
    Solving for the static endogenous variables of the model. This corresponds
    to Section 4.2 of Villemot (2011)

    Parameters
    ----------
    A: numpy ndarray
        the rotated Jacobian matrix of the model
    indexes : VariableIndexManager
        the index manager of the model
    gym: numpy ndarray
        the state response to backward-looking dynamic endogenous variables
    gyp: numpy ndarray
        the state response to forward-looking dynamic endogenous variables

    Returns
    -------
    np.ndarray
        the state response to static endogenous variables
    """
    ns = indexes.n_static

    Ap_cup = A[:ns, indexes.dr.forward_timed[1]]
    Am_cup = A[:ns, indexes.dr.backward_timed[-1]]
    A0s_cup = A[:ns, indexes.dr.static_timed[0]]

    # the indices are the Jacobian column indices corresponding to the
    #    dynamic variables, but we ensure they're in DR-order
    dr_llx = indexes.dr.llx
    _dj = dr_llx[1, ~(np.isnan(dr_llx[0, :]) & np.isnan(dr_llx[2, :]))]
    A0d_cup = A[:ns, _dj[~np.isnan(_dj)].astype(int)]

    # merging forward- and backward-looking dynamic response
    gyd = unp.array_shingle(gym, gyp, indexes.n_mixed)

    Agp = np.dot(np.dot(Ap_cup, gyp), gym)
    Agd = np.dot(A0d_cup, gyd)

    return -np.dot(np.linalg.inv(A0s_cup), Agp + Agd + Am_cup)


def solve_exo(
    A: np.ndarray,
    indexes: VariableIndexManager,
    gyp: np.ndarray,
    fu: np.ndarray
):
    """
    Solving for the response to exogenous variables of the model. This corresponds
    to Section  5 of Villemot (2011)

    Parameters
    ----------
    A: numpy ndarray
        the rotated Jacobian matrix of the model
    indexes : VariableIndexManager
        the index manager of the model
    gyp: numpy ndarray
        the state response to forward-looking dynamic endogenous variables
    fu : numpy ndarray
        the (unrotated) Jacobian column(s) corresponding to exogenous responses

    Returns
    -------
    np.ndarray
        the state response to exogenous variables
    """
    cont_static = A[:, indexes.dr.static_timed[0]]
    cont_backward = A[:, indexes.dr.backward_timed[0]]
    cont_forward = A[:, indexes.dr.pure_forward_timed[0]]

    Ap = A[:, indexes.dr.forward_timed[1]]
    implied_backward = np.dot(Ap, gyp) + cont_backward

    U = np.hstack((cont_static, implied_backward, cont_forward))
    return - np.linalg.solve(U, fu)
