"""
computing impulse responses of models
"""
from __future__ import annotations

from scipy import linalg
import numpy as np

from pynare.simul import simulate
from pynare.plotting import PathPlot
from pynare.core.variables import StochVar



class ImpulseResponse(object):

    def __init__(
        self,
        model: Model,
        exog: str,
        impulse: np.ndarray,
        size: Union[int, float]
    ):
        self.model = model
        self.exog = exog
        self.size = size

        self.impulse = impulse
        self.periods = impulse.shape[0]

    def __repr__(self):
        return f"ImpulseResponse(periods={self.periods}, size={self.size})"

    def plot(
        self,
        *args,
        **kwargs
    ):
        artist = PathPlot(self.impulse, self.model.endog)
        kwargs.setdefault('layout', 'rect')

        return artist.plot(*args, **kwargs)



def impulse_response(
    model: Model,
    shock: Union[Sequence[int, str], int, str] = 0,
    periods: int = 20,
    size: Union[int, float] = 1
):
    """
    compute the impulse response of a model to an exogenous shock

    Parameters
    ----------
    model : Model
        the model to compute an impulse response of
    shock : int | str | Iterable[int, str] ( = 0 )
        the exogenous variable to shock. if an int, the 'shock'-th exogenous
        variable is shocked, and if a str, 'shock' is interpreted as the name of
        the exogenous variable to shock
    periods : int ( = 20 )
        the number of periods to calculate the responses for
    size : int | float ( = 1 )
        the size of the shock (in standard deviation space)

    Returns
    -------
    ImpulseResponse
    """
    # check that `shock` is an exogenous variable in the model
    if isinstance(shock, StochVar):
        try:
            stoch_index = np.where(model.stoch.names == shock.name)[0].item()
            stoch = shock
        except ValueError:
            # thrown by `item()` when `where()[0]` evaluates to empty array
            raise ValueError(
                f"'{shock}' is not a stochastic variable of the model"
            ) from None

    elif isinstance(shock, int):
        try:
            stoch_index = shock # used after the `if` blocks
            stoch = model.stoch[stoch_index]
        except IndexError:
            n_exog = len(model.stoch)
            raise ValueError(
                f"shock index: {shock}. there are only {n_exog} stochastic vars"
            ) from None

    elif isinstance(shock, str):
        try:
            stoch_index = np.where(model.stoch.names == shock)[0].item()
            stoch = model.stoch[stoch_index]
        except ValueError:
            raise ValueError(
                f"'{shock}' is not a stochastic variable of the model"
            ) from None

    else:
        raise TypeError(f"{type(shock)}. `shock` can only be int or str, or StochVar")

    # generate one-period shock of size `size` (in standard deviation space)
    n_stochs = len(model.stoch)
    shock = np.zeros((periods, n_stochs), dtype=float)
    shock[0, stoch_index] = size

    # run simulation with single impulse
    sim = simulate(model, shocks=shock)
    paths = sim.paths
    impulse = paths - model.ss.values

    response = ImpulseResponse(model, exog=stoch, impulse=impulse, size=size)
    return response
