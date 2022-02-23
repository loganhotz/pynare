"""
computing moments of model processes
"""
from __future__ import annotations

import numpy as np
from scipy import linalg

import pynare.utils.numpy as unp



def mean(md: Model):
    """
    compute the average values of a model's observable variables. adjustments are
    made to models solved at second-order, following Kim, Kim, Schaumburg, and Sims
    (2008) "Calculating and Using Second-Order Accurate Solutions of Discrete Time
    Dynamic Equilibrium Models", although I found Andreasen, Fernandez-Villaverde,
    and Rubio-Ramirez (2017), "The Pruned State-Space System for Non-Linear DSGE
    Models: Theory and Empirical Applications" to be a bit clearer

    Parameters
    ----------
    md : Model
        a pynare Model

    Returns
    -------
    avg : np.ndarray
        a vector of length len(md.endog)
    """
    ss = md.ss.values
    if md.solution_order == 1:
        return ss

    elif md.solution_order == 2:
        # E[x_t^2] = Var[x], and first-order transition array
        var_state = md.statespace.sigma
        g_xtp1x, _ = md.statespace.kalman_transition

        # second-order transition arrays. only select those determining state evolution
        dr_sidx = md.indexes.dr.state
        del2, ghxx, ghuu = md.solution.del2, md.solution.ghxx, md.solution.ghuu

        # x_{t+1} = g_{xt+1, x}*x_t + g_{ut+1}*u_{t+1}
        #               + (1/2) * (del2 + G_{x,x}*x_t^2 + G_{u,u}*u_t^2
        # yields E[x] = g_{xt+1,x}*E[x] + (1/2)*(del2 + G_{x,x}*Var[x] + G_{u,u}*Var[u]
        second_order = (del2[dr_sidx] + np.dot(ghxx[dr_sidx], var_state.flatten()) \
            + np.dot(ghuu[dr_sidx], md.sigma.flatten()))/2
        expt_x = linalg.solve(np.eye(*g_xtp1x.shape) - g_xtp1x, second_order)

        # calculate mean adjustment for observables & make them declaration order
        adj_mean_inv = unp.matmul_scalar(md.solution.lower_solution.gy, expt_x) \
            + ( del2 + unp.matmul_scalar(ghxx, var_state.flatten()) \
            + unp.matmul_scalar(ghuu, md.sigma.flatten()) ) / 2
        adj_mean = adj_mean_inv[md.indexes.dc_order]

        return ss + adj_mean

    else:
        raise NotImplementedError(
            "can only calculate means for 1st- and 2nd-order solutions"
        ) from None



def var(md: Model):
    """
    compute the variance of a model's observable variables

    Parameters
    ----------
    md : Model
        a pynare Model

    Returns
    -------
    var : np.ndarray
        a vector of length len(md.endog)
    """
    return np.diag(cov(md))



def std(md: Model):
    """
    compute the standard deviation of a model's observable variables

    Parameters
    ----------
    md : Model
        a pynare Model

    Returns
    -------
    std : np.ndarray
        a vector of length len(md.endog)
    """
    return np.sqrt(var(md))



def corr(md: Model):
    """
    compute the contemporaneous correlation array of a model's observable variables

    Parameters
    ----------
    md : Model
        a pynare Model

    Returns
    -------
    corr : np.ndarray
        a square array of size (en, en) where en = len(md.endog)
    """
    cov_arr = cov(md)
    std_dev = np.sqrt(np.diag(cov_arr))

    # std devs. to translate from autocovariance to autocorrelation
    scale = np.outer(std_dev, std_dev)
    return np.divide(cov_arr, scale)



def cov(md: Model):
    """
    compute the variance-covariance array of a model's observable variables

    Parameters
    ----------
    md : Model
        a pynare Model

    Returns
    -------
    cov : np.ndarray
        a square array of size (en, en) where en = len(md.endog)
    """
    if md.solution_order == 1:
        gy, gu = md.solution.gy, md.solution.gu

    elif md.solution_order == 2:
        gy = md.solution.lower_solution.gy
        gu = md.solution.lower_solution.gu

    else:
        raise NotImplementedError(
            "can only calculate covariances for 1st- and 2nd-order solutions"
        ) from None

    # translate to declaration order
    gy_dc_, gu_dc_ = gy[md.indexes.dc_order], gu[md.indexes.dc_order]
    gy_dc, gu_dc = unp.ensure_2darray((gy_dc_, gu_dc_))

    # variance matrix of the state-space transition process
    var_state = md.statespace.sigma

    # from observation equation y_t = h(x, u), recover variance
    x_term = np.matmul(gy_dc, np.matmul(var_state, np.transpose(gy_dc)))
    u_term = np.matmul(gu_dc, np.matmul(md.sigma, np.transpose(gu_dc)))
    return x_term + u_term



def autocorr(
    md: Model,
    p: int = 5
):
    """
    compute the autocorrelation of a model's variables, assuming an AR(p) process

    Parameters
    ----------
    md : Model
        a pynare model
    p : int ( = 5 )
        the number of periods for which to compute the autocorrelation

    Returns
    -------
    autocorr : np.ndarray
        (p, n_endog, n_endog) array, where autocorr[i, j, k] is the i-period correlation
        of the j- and k-th endogenous variables (in declaration order)
    """
    acv = autocov(md, p)

    # std devs. to translate from autocovariance to autocorrelation
    std_dev = std(md)
    scale = np.outer(std_dev, std_dev)

    with np.errstate(divide='ignore', invalid='ignore'):
        # locally silence RuntimeWarnings about dividing by zero or infinity
        return np.divide(acv, scale)



def autocov(
    md: Model,
    p: int = 5
):
    """
    compute the autocovariance of a model's variables, assuming an AR(p) process

    Parameters
    ----------
    md : Model
        a pynare model
    p : int ( = 5 )
        the number of periods for which to compute the autocovariance

    Returns
    -------
    autocov : np.ndarray
        (p, n_endog, n_endog) array, where autocov[i, j, k] is the i-period covariance
        of the j- and k-th endogenous variables (in declaration order)
    """
    if md.solution_order == 1:
        gy, gu = md.solution.gy, md.solution.gu

    elif md.solution_order == 2:
        gy = md.solution.lower_solution.gy
        gu = md.solution.lower_solution.gu

    else:
        raise NotImplementedError(
            "can only calculate autocovariances for 1st- and 2nd-order solutions"
        ) from None

    # E[x_t^2] = Var[x], and first-order transition arrays
    var_state = md.statespace.sigma
    g_xtp1x, g_utp1 = md.statespace.kalman_transition

    # translate to declaration order
    gy_dc_, gu_dc_ = gy[md.indexes.dc_order], gu[md.indexes.dc_order]
    gy_dc, gu_dc = unp.ensure_2darray((gy_dc_, gu_dc_))

    # from observation equation y_t = h(x, u), create observable -> state var array
    x_term = np.matmul(g_xtp1x, np.matmul(var_state, np.transpose(gy_dc)))
    u_term = np.matmul(g_utp1, np.matmul(md.sigma, np.transpose(gu_dc)))
    obs_state = x_term + u_term

    n_endog = gy_dc.shape[0]
    acv = np.empty((p, n_endog, n_endog), dtype=float)
    for i in range(p):
        acv[i, :, :] = np.matmul(gy_dc, obs_state)
        obs_state = np.matmul(g_xtp1x, obs_state)

    return acv
