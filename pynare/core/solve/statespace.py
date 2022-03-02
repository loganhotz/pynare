"""
module for the state space representation of a model
"""
from __future__ import annotations

import numpy as np
from scipy import linalg
from collections import namedtuple
from functools import cached_property

import pynare.utils.numpy as unp
from pynare.errors import ModelIdentificationError



class StateSpace(object):

    def __init__(self, model: Model):
        self.model = model

        # would like to move all of these arrays to cached properties at some
        #   point, since a new StateSpace is created each time an update is made
        D, E, A = _compute_structural_state_space(self.model)

        self.jacobian = A
        self.lhs = D
        self.rhs = E

        # solve state space equation; partition according to explosive eigens
        eigs, Z_11, Z_12, Z_21, Z_22, T_11, T_12, T_22, S_11, S_12, S_22 = \
            unp.partitioned_qz(D, E)
        self.eigen = eigs

        self.larrays = LHSArrays(T_11, T_12, T_22)
        self.rarrays = RHSArrays(S_11, S_12, S_22)
        self.zarrays = ZArrays(Z_11, Z_12, Z_21, Z_22)

        # indicator for whether or not BK conditions have been checked. if `None`,
        #   those conditions have not been checked
        self.is_verified = None


    def verify(self):
        """
        front-facing method for checking whether or not the BK conditions are met
        by this model specification
        """
        if self.is_verified is None:
            self._run_verification()
        return None


    def _run_verification(self):
        """
        checking the Blanchard-Kahn (1980) conditions
        """
        abs_eigs = np.absolute(self.eigen)
        n_explosive = np.sum((abs_eigs > 1) | np.isinf(abs_eigs))
        n_forward = self.model.indexes.n_forward

        if n_explosive != n_forward:
            self.is_verified = False
            raise ModelIdentificationError('order', n_explosive, n_forward)

        Z22 = self.zarrays.Z_22
        if np.linalg.matrix_rank(Z22) != Z22.shape[0]:
            self.is_verified = False
            raise ModelIdentificationError('rank')

        self.is_verified = True
        return None


    @cached_property
    def sigma(self):
        """
        from the first-order state variable transition process
            x_{t+1} = g_{xt+1, x}*x_t + g_{ut+1}*u_{t+1},
        recover the variance matrix of the vector of state variables:
            Var[x] = g_{xt+1, x}*Var[x]*g_{xt+1, x}^T + g_{ut+1}*Var[u]*g_{ut+1}^T
        """
        self.verify()
        g_xtp1x, g_utp1 = self.kalman_transition

        # the variance equation above is a discrete lyapunov: X - A*X*A^T - Q = 0
        Q = np.matmul(g_utp1, np.matmul(self.model.sigma, np.transpose(g_utp1)))
        return linalg.solve_discrete_lyapunov(g_xtp1x, Q)

    @cached_property
    def kalman_transition(self):
        """
        create the Kalman transition matrices of the state-space
        """
        self.verify()
        sol = self.model.first_order_solution

        dr_sidx = self.model.indexes.dr.state
        n_state = len(dr_sidx)

        A = np.zeros((n_state, n_state), dtype=float)
        A[:, :n_state] = sol.gy[dr_sidx]

        B = sol.gu[dr_sidx]

        return unp.ensure_2darray((A, B))

    def __repr__(self):
        return 'StateSpace'



# named arrays in state-space representation
LHSArrays = namedtuple('LHSArrays', ('T_11', 'T_12', 'T_22'))
RHSArrays = namedtuple('RHSArrays', ('S_11', 'S_12', 'S_22'))
ZArrays = namedtuple('ZArrays', ('Z_11', 'Z_12', 'Z_21', 'Z_22'))



def _compute_structural_state_space(model: Model):
    """
    compute the structural state space representation of a model. specifically, this
    method computes the `D` and `E` matrices in Equation 8 of Villemot (2011)

    Parameters
    ----------
    model : Model
        a pynare model

    Returns
    -------
    state_space : 3-tuple of np.ndarrays
        the `D` and `E` matrices are returned, along with the rotated jacobian matrix
    """
    indexes = model.indexes

    J = model.jacobian.array
    S = J[:, indexes.static_timed[0]]

    Q, R = np.linalg.qr(S, mode='complete')

    # for model to be properly identified, rank(R) must equal number of static vars
    _r = np.linalg.matrix_rank(R)
    if _r != indexes.n_static:
        raise ModelIdentificationError(_r, indexes.n_static)

    # rotated jacobian
    A = np.dot(np.transpose(Q), J)

    # partition as according to section 4.1 of Villemot (2011)
    nd = indexes.n_dynamic
    Ap_tilde = A[-nd:, indexes.dr.forward_timed[1]]
    Am_tilde = A[-nd:, indexes.dr.backward_timed[-1]]
    A0p_tilde = A[-nd:, indexes.dr.forward_timed[0]]

    # model might not have any backward variables
    A0m_tilde = A[-nd:, indexes.dr.pure_backward_timed[0]]
    if A0m_tilde.size == 0:
        A0m_tilde = np.zeros((nd, 1))

    # set up the structural state space representation
    nm = indexes.n_mixed
    if nm > 0:
        # mixed variables are present
        nb, nf = indexes.n_backward, indexes.n_forward

        # in section 4.1, two definitions of A0m_tilde and Ap_tilde are given - we
        #   use the first, which means A0m_tilde needs to be augmented with zeros if
        #   n_endo - (n_forward + n_static) > 0
        n_diff = indexes.n_endo - (nf + indexes.n_static)

        if n_diff:
            zeros = np.zeros((nd, n_diff), dtype=float)
            A0m_tilde = np.hstack((A0m_tilde, zeros))

            Im = np.eye(nm, nb, k=n_diff)

        else:
            Im = np.eye(nm, nb)

        Ip = np.eye(nm, nf)

        D = np.block([[A0m_tilde, Ap_tilde], [Im, np.zeros((nm, nf))]])
        E = np.block([[-Am_tilde, -A0p_tilde], [np.zeros((nm, nb)), Ip]])

    else:
        D = np.hstack((A0m_tilde, Ap_tilde))
        E = - np.hstack((Am_tilde, A0p_tilde))

    return D, E, A
