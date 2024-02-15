"""
module for Kalman and (eventually) particle filters of models
"""
from __future__ import annotations

import warnings

import pandas as pd
import numpy as np
import scipy.linalg as linalg

from functools import cached_property

from pynare.stats.observation import ObservedModel
import pynare.utils.numpy as unp



from rich import print


class ModelFilter(object):

    def __init__(
        self,
        model: Model,
        data: np.ndarray,
        obs: dict[str, str] = {}
    ):
        if isinstance(model, ObservedModel):
            if obs:
                msg = (
                    "given model already has observations incorporated. ignoring "
                    "passed 'obs' dictionary"
                )
                warnings.warn(msg)

            self.model = model

        else:
            self.model = ObservedModel(model=model, exprs=obs)

        # save observed variable names to ensure they match the passed data names
        self.obs = self.model.obs

        # strip indices and column labels, leaving just column-oriented array
        arr, labels = _preprocess_data(data)
        self.data = arr

        if labels:
            if len(labels) != len(self.obs):
                raise ValueError(
                    f"number of observed vars ({len(self.obs)} does not match number "
                    f"number of data series ({len(labels)})."
                )

            for lb in labels:
                if lb not in self.obs:
                    raise NameError(f"{repr(lb)} is not an observed var")

            # reordering data if columns are different from observed vars decl order
            idx = self.obs.get_loc(labels)
            self.data = self.data[:, idx]

        else:
            if self.data.shape[1] != len(self.obs):
                raise ValueError(
                    f"number of observed vars ({len(self.obs)} does not match number "
                    f"number of data series ({self.data.shape[1]})."
                )

    def __repr__(self):
        class_ = type(self).__name__
        return f"{class_}({repr(self.model)})"



class KalmanFilter(ModelFilter):
    """
    class that applies a Kalman filter to a model, provided a dataset. mathematical
    notation mostly follows Durbin and Koopman, "Time Series Analysis by State Space
    Methods", except we use `x` to denote the state vector, and they use `alpha`
    """

    def __init__(
        self,
        model: Model,
        data: np.ndarray,
        obs: dict[str, str] = {},
        init: str = ''
    ):
        # set model, data, meas, and meas_idx attributes
        super().__init__(model=model, data=data, obs=obs)

        if init:
            raise NotImplementedError(f"initialization scheme: '{init}'")
        self.init = init


    @cached_property
    def indices(self):
        """prepare the indices for filtering"""
        dr_sidx, midx = self.model.indexes.dr.state, self.meas_idx

        # 'dr' attribute enforces order in state vars; measured variables are not
        #   guaranteed to be, though
        tidx = np.sort(np.hstack((dr_sidx, midx)))
        st_locs = np.searchsorted(tidx, dr_sidx)
        m_locs = np.searchsorted(tidx, np.sort(midx))

        return tidx, st_locs, m_locs


    @cached_property
    def transition(self):
        """returns the arrays in the transition equation of the filter"""
        tidx, st_locs, _ = self.indices

        sol = self.model.solution
        gy, gu = unp.ensure_2darray((sol.gy, sol.gu))

        T = np.zeros((len(tidx), len(tidx)), dtype=float)
        T[:, st_locs] = gy[tidx]
        R = gu[tidx]

        return T, R


    @property
    def init_state(self):

        if self.init:
            raise NotImplementedError(self.init)
        else:
            tidx, _, _ = self.indices
            state_meas = np.zeros(len(tidx), dtype=float)

            T, R = self.transition
            Q = np.matmul(R, np.matmul(self.model.sigma, np.transpose(R)))

            P = linalg.solve_discrete_lyapunov(T, Q)
            return state_meas, P

    def loglike(self):

        sol = self.model.solution
        T, R, Z = sol.arrays

        # construct initial conditions for filter
        Q = np.matmul(R, np.matmul(self.model.sigma, np.transpose(R)))
        P = linalg.solve_discrete_lyapunov(T, Q)

        n_state = len(T)
        init_state = np.zeros(n_state, dtype=float)

        return _kalman_filter_loglikelihood(
            data=self.data,
            x0=init_state,
            P0=P,
            T=T, R=R, Q=Q, Z=Z
        )


    def filter(self):

        x, P = self.init_state
        T, R = self.transition
        Q = np.matmul(R, np.matmul(self.model.sigma, np.transpose(R)))

        _, _, m_locs = self.indices
        return _simple_kalman_transition_selection(
            data=self.data,
            x0=x, P0=P,
            T=T, Q=Q,
            locs=m_locs
        )



def filter_model(
    model: Model,
    data: Union[np.array, pd.DataFrame],
    obs: dict[str, str] = {},
    kind: str = 'kalman',
    init: str = '' # initialization scheme
):
    """
    """

    if kind == 'kalman':
        return KalmanFilter(model, data, obs, init)

    else:
        raise NotImplementedError(f"'kind' = {repr(kind)}. only Kalman filters allowed")



def _kalman_filter_loglikelihood(
    data: np.ndarray,
    x0: np.ndarray,
    P0: np.ndarray,
    T: np.ndarray,
    R: np.ndarray,
    Q: np.ndarray,
    Z: np.ndarray,
    start: int = 0
):
    """
    compute Kalman Filter log likelihood from a model's state-space representation:
            x_t = T x_{t-1} + R eta_t,    eta_t ~ N(0, Q)
            y_t = Z x_t
    """

    llk = 0
    pi2 = np.log(2*np.pi)

    xt, Pt = x0, P0
    n_obs, n_vars = data.shape

    print(xt)
    print(Z)
    print(data[0])

    for i in range(n_obs):
        vt = data[i] - np.dot(Z, xt)

        # commonly used P*Z^T; ensure Ft is symmetric, as it should be
        PtZh = np.matmul(Pt, np.transpose(Z))
        Ft = np.matmul(Z, PtZh)
        Ft = (Ft + Ft.T) / 2

        if np.any(Ft < 0):
            print(i)
            print(Ft)
            print(linalg.det(Ft))
            print('')

        # log-likelihood terms
        det_Ft = np.log(linalg.det(Ft))
        invFt_vt = linalg.solve(Ft, vt, assume_a='sym')

        if i >= start:
            llk -= ( n_vars * pi2 + det_Ft + np.dot(vt, invFt_vt) ) / 2

        xtt = xt + np.matmul(PtZh, invFt_vt)
        xt = np.dot(T, xtt)
        # print(vt)


def _simple_kalman_transition_selection(
    data: np.ndarray,
    x0: np.ndarray,
    P0: np.ndarray,
    T: np.ndarray,
    Q: np.ndarary,
    locs: np.ndarray
):
    """
    using a model's first-order relations, filter with the provided data, making
    the assumption that the forecast errors are simple affine functions of entries
    in the state vector, rather than linear combinations of those entries.
        the filtering process is the usual Kalman setup, mostly using notation from
    Durbin and Koopman:
            x_t = T*x_tm1 + e_t,        e_t ~ N(0, Q)
            y_t = x_t[locs]
    where the `y` variables have observed data and `x` contains the state variables
    that are mapped into the measurement equation
    """

    llk = 0
    x, P = x0, P0

    for i in range(data.shape[0]):
        vt = data[i] - x[locs]
        Ft = P[np.ix_(locs, locs)]
        Ft_inv = unp.inv_sym_pd(Ft)

        PFi = np.matmul(P[:, locs], Ft_inv)
        xtt = x + np.dot(PFi, vt)
        Ptt = P - np.matmul(PFi, P[locs, :])

        x = np.dot(T, xtt)
        P = np.matmul(T, np.matmul(Ptt, np.transpose(T))) + Q

        llk_ = np.log(linalg.det(Ft))
        llk += llk_

    return llk



def _preprocess_data(
    data: Union[np.ndarray, pd.DataFrame]
):
    """
    prepare a dataset to be used as a filter. strips indexes of pandas objects,
    and assumes column labels are the measured variables' names

    Returns
    -------
    arr, endog_names
    """
    try:
        # covering case where `data` is a pandas Series
        data = data.to_frame()
    except AttributeError:
        pass

    try:
        endog = data.columns
        return data.to_numpy(), endog.tolist()

    except AttributeError:
        # `data` is already a numpy array, or compatible with numpy arrays
        return np.asarray(data), []
