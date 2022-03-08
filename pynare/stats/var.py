"""
computing vector-autoregression properties of a model
"""
from __future__ import annotations

import numpy as np
from scipy import linalg

import pynare.utils.numpy as unp
from pynare.utils.dtypes import ensure_list



def var_decomp(
    md: Model,
    endog: Sequence[str] = None,
    stoch: Sequence[str] = None
):
    """
    compute the variance decomposition of a model. uses a cholesky decomposition

    Parameters
    ----------
    md : Model
        a pynare Model
    endog : Sequence[str] ( = None )
        a sequence of endogenous variable names. only those variables whose names
        are included will have their variance decomposed
    stoch : Sequence[str] ( = None )
        a sequence of stochastic variable names. only those variables whose names
        are included will be displayed in the variance decomposition

    Returns
    -------
    decomp : np.ndarray
        an (n_endog, n_stoch) array whose (i, j)-th entry is the percent of variance
        in endog `i` that is attributable to exog `j`, in declaration order
    """

    if len(md.stoch) == 1:
        # quick check for models with only one stochastic variable
        decomp_shares = np.ones((len(md.endog), 1), dtype=float)

        if endog:
            endog_names = md.endog.names
            endog_idx = []

            for eg in ensure_list(endog):
                try:
                    endog_idx.append(np.where(endog_names == eg)[0].item())
                except ValueError:
                    raise ValueError(f"'{eg}' is not an endogenous variable") from None
            return decomp_shares[endog_idx]

        return decomp_shares

    # this is 'None' if the model hasn't been solved yet
    if not md.solution_order:
        _ = md.solve()

    if md.solution_order == 1:
        gy, gu = md.solution.gy, md.solution.gu

    elif md.solution_order == 2:
        gy = md.solution.lower_solution.gy
        gu = md.solution.lower_solution.gu

    else:
        raise NotImplementedError(
            "can only calculate variance decomps for 1st- and 2nd-order solutions"
        ) from None

    # prepare transition arrays
    gy, gu = unp.ensure_2darray((gy, gu))
    gy_dc, gu_dc = gy[md.indexes.dc_order], gu[md.indexes.dc_order, :]
    g_xtp1x, g_utp1 = md.statespace.kalman_transition

    # perturb and decompose variance matrix
    decomp_tol = 1e-12
    perturb_sigma = md.sigma + decomp_tol*np.eye(*md.sigma.shape, dtype=float)
    sig_chol = linalg.cholesky(perturb_sigma, lower=True)

    state_chol = np.matmul(g_utp1, sig_chol)
    gu_chol = np.matmul(gu_dc, sig_chol)

    # using state-space transition equation to back out state vector's variance
    sigma_gu = np.matmul(state_chol, np.transpose(state_chol))
    var_state = linalg.solve_discrete_lyapunov(g_xtp1x, sigma_gu)

    # pre-allocate decomposition array
    n_endog, n_stoch = len(md.endog), len(md.stoch)
    decomp = np.empty((n_endog, n_stoch), dtype=float)

    # calculate state-space transition and observation equations for each stoch
    for i in range(n_stoch):
        var_stochi = np.outer(state_chol[:, i], state_chol[:, i])
        var_statei = linalg.solve_discrete_lyapunov(g_xtp1x, var_stochi)

        xi_term = np.matmul(gy_dc, np.matmul(var_statei, np.transpose(gy_dc)))
        ui_term = np.outer(gu_chol[:, i], gu_chol[:, i])
        decomp[:, i]  = np.abs(np.diag(xi_term + ui_term))

    # equivalent to diagonal(gy_dc*var_state*gy_dc^T + gu_chol*gu_chol^T)
    total_variance = np.sum(decomp, axis=1)

    with np.errstate(divide='ignore', invalid='ignore'):
        # locally silence RuntimeWarnings about dividing by zero or infinity
        decomp_shares = np.divide(decomp, total_variance[:, np.newaxis])

    # selecting portions of variance decomposition array
    if endog or stoch:
        endog_names, stoch_names = md.endog.names, md.stoch.names

        if endog:
            endog_idx = []
            for eg in ensure_list(endog):
                try:
                    endog_idx.append(np.where(endog_names == eg)[0].item())
                except ValueError:
                    raise ValueError(f"'{eg}' is not an endogenous variable") from None

        else:
            endog_idx = []

        if stoch:
            stoch_idx = []
            for sc in ensure_list(stoch):
                try:
                    stoch_idx.append(np.where(stoch_names == sc)[0].item())
                except ValueError:
                    raise ValueError(f"'{sc}' is not a stochastic variable") from None

        else:
            stoch_idx = []

        if endog_idx and stoch_idx:
            return decomp_shares[np.ix_(endog_idx, stoch_idx)]

        elif endog_idx:
            return decomp_shares[endog_idx]

        elif stoch_idx:
            return decomp_shares[:, stoch_idx]

    else:
        return decomp_shares
