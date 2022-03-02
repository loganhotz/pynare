"""
performing simulations of pynare models
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import linalg

import pynare.utils.numpy as unp
from pynare.core.solve.base import LinearSolution
from pynare.plotting import PathPlot



class ModelSimulation(object):

    def __init__(
        self,
        model: Model,
        paths: np.ndarray,
        shocks: np.ndarray
    ):
        self.model = model

        self.paths = paths
        self.shocks = shocks
        self.periods = paths.shape[0]


    def plot(
        self,
        *args,
        **kwargs
    ):
        artist = PathPlot(self.paths, self.model.endog)
        kwargs.setdefault('layout', 'rect')

        return artist.plot(*args, **kwargs)


    def __repr__(self):
        return f"ModelSimulation(periods={self.periods})"


    def corr(
        self,
        drop: int = 0
    ):
        """
        compute the correlation matrix for the simulated paths of the endogenous
        variables.

        Parameters
        ----------
        drop : int ( = 0 )
            the number of periods to drop before calculating the empirical correlation

        Returns
        -------
        pandas DataFrame
            an (n, n) DataFrame, where `n` is the number of endogenous variables
        """

        # drop and de-mean
        paths = _drop_simulated_paths(self.paths, drop)
        centered = paths - np.mean(paths, axis=0)

        # compute covariance matrix, then correlation matrix
        cov = np.matmul(np.transpose(centered), centered) / paths.shape[0]
        std = np.std(paths, axis=0)
        corr = np.divide(cov, np.outer(std, std))

        endog = self.model.endog.names
        return pd.DataFrame(corr, index=endog, columns=endog)


    def autocorr(
        self,
        p: int = 5,
        drop: int = 0
    ):
        """
        compute the autocorrelation coefficients for the simulated paths of the
        endogenous variables. the first `p`-th coefficients are computed and
        reported.

        Parameters
        ----------
        p : int ( = 5)
            the order of autocorrelation coefficients to print
        drop : int ( = 0 )
            the number of periods to drop before calculating the empirical mean and
            variance of the simulated paths

        Returns
        -------
        pandas DataFrame
            (i, j)-th entry is the j-th autocorrelation coefficient of endogenous
            variable i (in declaration order)
        """

        # drop requested number of periods
        paths = _drop_simulated_paths(self.paths, drop)
        periods = paths.shape[0]

        # check there are enough periods to compute p-th autocorrelation
        if p > periods:
            raise ValueError(
                "too many autocorrelation lags given this many periods: "
                f"{p} > {periods}"
            )

        # compute empirical moments of simulated data
        centered = paths - np.mean(paths, axis=0)
        variance = np.var(paths, axis=0)

        # pre-allocate then compute autocorrelation
        arr = np.full((paths.shape[1], p), np.nan)
        for k in range(1, p+1):
            X_t = centered[k:, :]
            X_tmk = centered[:-k, :]

            ssq = np.einsum('ij, ij->j', X_t, X_tmk) / (periods - k)
            arr[:, k-1] = np.divide(ssq, variance)

        return pd.DataFrame(
            arr,
            index=self.model.endog.names,
            columns=[f"k_{k+1}" for k in range(p)]
        )



class LinearSimulation(ModelSimulation):

    def __init__(
        self,
        model: Model,
        order: int,
        paths: np.ndarray,
        shocks: np.ndarray
    ):
        super().__init__(model, paths, shocks)
        self.order = order

    def __repr__(self):
        return f"LinearSimulation(order={self.order}, periods={self.periods})"



class ModelSimulator(object):

    def __init__(self, model):
        self.model = model
        self.model.update()



class LinearSimulator(ModelSimulator):

    def __init__(self, model):
        super().__init__(model)

        # locating the t-1 state variables in dr order
        self.dr_order = self.model.indexes.dr_order

        lag_state_maybe_scalar = self.model.indexes.dr.state
        try:
            self.lag_state = lag_state_maybe_scalar.item()
        except ValueError:
            self.lag_state = lag_state_maybe_scalar

    def _simulate_first_order(
        self,
        shocks: np.ndarray,
        init: np.ndarray
    ):
        """
        simulate a model using its first-order solution
        """
        T = shocks.shape[0]

        # paths of the endogenous variables
        ys = self.model.ss.values
        paths = np.zeros((T, len(ys)), dtype=float)
        paths[0, :] = init - ys

        # policy matrices and innovations to endogenous variables
        gy, gu = self.model.solution.gy, self.model.solution.gu
        gu_arr = unp.ensure_2darray(gu)
        innovs = np.matmul(shocks, np.transpose(gu_arr))

        # index management
        dr_order = self.dr_order
        dr_state = dr_order[self.lag_state]

        if gu.size == 0:
            raise NotImplementedError("cannot simulate deterministic models yet")
        elif gy.size == 0:
            raise NotImplementedError("can't simulate purely forward-looking models yet")
        else:
            for t in range(1, T):
                yhat = paths[t-1, dr_state]
                paths[t, dr_order] = np.dot(gy, yhat) + innovs[t-1, :]

        # move from deviation space to level space
        paths = paths + ys
        return LinearSimulation(self.model, 1, paths, shocks)

    def _simulate_second_order(
        self,
        shocks: np.ndarray,
        init: np.ndarray
    ):
        """
        simulate a model using its second-order solution
        """
        T = shocks.shape[0]

        # paths of the endogenous variables
        ys = self.model.ss.values
        paths = np.zeros((T, len(ys)), dtype=float)
        paths[0, :] = init

        # policy matrices
        sol = self.model.solution
        first_sol = sol.lower_solution
        ghxx, ghxu, ghuu, del2 = unp.ensure_2darray(
            (sol.ghxx/2, sol.ghxu, sol.ghuu/2, sol.del2/2))
        gy, gu = first_sol.gy, first_sol.gu

        # index management. use array of indices because of kronecker products
        dr_order = self.dr_order
        lag = [self.lag_state] if isinstance(self.lag_state, int) else self.lag_state
        dr_state = dr_order[lag]

        const = del2.squeeze() + ys[dr_order]
        for t in range(1, T):
            yhat = paths[t-1, dr_state] - ys[dr_state]
            ut = shocks[t-1, :]

            # first-order terms
            dy = unp.matmul_scalar(gy, yhat)
            du = unp.matmul_scalar(gu, ut)

            # second-order terms
            dydy = unp.matrix_kronecker_product(ghxx, yhat)
            dudu = unp.matrix_kronecker_product(ghuu, ut)
            dydu = unp.matrix_kronecker_product(ghxu, yhat, ut)

            paths[t, dr_order] = const + dy + du + dydy + dudu + dydu

        return LinearSimulation(self.model, 2, paths, shocks)

    def simulate(
        self,
        shocks: Union[Any, np.ndarray],
        periods: int,
        init: Union[Any, np.ndarray]
    ):
        shock_paths = _generate_simul_shocks(self.model, shocks, periods)

        if init is None:
            init = self.model.ss.values

        order = self.model.solution.order
        if order == 1:
            return self._simulate_first_order(shock_paths, init)
        elif order == 2:
            return self._simulate_second_order(shock_paths, init)
        else:
            raise NotImplementedError("cannot simulate linear solutions of order > 2")



def simulate(
    model: Model,
    shocks: np.ndarray = None,
    periods: int = 200,
    init: np.ndarray = None,
    order: int = 0,
):
    """
    simulate a model

    Parameters
    ----------
    model : Model
        the pynare model to simulate. if a solution has not already been computed
        and `order` is not provided, the default linearization order is used
    shocks : numpy ndarray ( = None )
        the paths of innovations to all the exogenous variables to use in the
        simulation, in percent deviation space. if not provided, the shock paths
        are random normal
    periods : int ( = 200 )
        the number of periods to run the simulation for. if `shocks` is provided,
        this parameter is ignored
    init : np.ndarray ( = None )
        initial values for the endogenous variables in declaration order. if not
        provided, the steady state is used
    order : int ( = 0 )
        the order to solve the model at. if not provided, use existing solution
        if available, or the default order if not. if provided, overwrite existing
        solution

    Returns
    -------
    ModelSimulation
    """

    if order:
        # if `order` is provided, solve (or resolve) at that order
        solution = model.solve(order=order)
    else:
        # if no `order`, use existing solution if there is one. otherwise, default
        try:
            solution = model._solution
        except AttributeError:
            solution = model.solve()

    if isinstance(solution, LinearSolution):
        simulator = LinearSimulator(model)
        return simulator.simulate(periods=periods, shocks=shocks, init=init)

    else:
        raise NotImplementedError("non-linear solutions cannot be simulated")



def _generate_simul_shocks(
    model: Model,
    shocks: np.ndarray,
    periods: int,
    distribution: str = 'normal'
):
    """
    create an array of innovations to exogenous variables based on the covariate
    matrix of the model's shocks and the number of periods
    """

    corr_struct = linalg.cholesky(model.sigma)

    if shocks is None:
        n_exog = len(model.stoch)

        if distribution == 'normal':
            noise = np.random.normal(size=(periods, n_exog))
            return np.matmul(noise, corr_struct)

        else:
            raise NotImplementedError("non-normal distributions cannot be simulated")

    else:
        shock_arr = unp.ensure_2darray(shocks)
        return np.matmul(shock_arr, corr_struct)


def _drop_simulated_paths(
    paths: np.ndarray,
    drop: int
):
    """
    remove the first `drop` periods from an array of simulated paths

    Parameters
    ----------
    paths : np.ndarray
        an (N, n_endog) array
    drop : int
        the number of periods to drop from beginning of paths

    Returns
    -------
    truncated : np.ndarray
        (N-drop, n_endog) array
    """
    if drop < 0:
        raise ValueError(f"if provided, `drop` must be greater than zero: {drop}")

    periods = paths.shape[0]
    if drop > periods:
        raise ValueError(
            "the number of dropped periods exceeds the number of simulated "
            f"periods: {drop} > {periods}"
        )

    # drop requested number of periods
    return paths[drop:, :]
