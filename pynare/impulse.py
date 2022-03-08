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
    exog: Union[Sequence[int, str], int, str] = 0,
    periods: int = 20,
    size: Union[int, float] = 1
):
    """
    compute the impulse response of a model to an exogenous shock

    Parameters
    ----------
    model : Model
        the model to compute an impulse response of
    exog : int | str | Iterable[int, str] ( = 0 )
        the exogenous variable to shock. if an int, the 'exog'-th exogenous
        variable is shocked, and if a str, 'exog' is interpreted as the name of
        the exogenous variable to shock
    periods : int ( = 20 )
        the number of periods to calculate the responses for
    size : int | float ( = 1 )
        the size of the shock (in standard deviation space)

    Returns
    -------
    ImpulseResponse
    """

    if isinstance(exog, int):
        try:
            exog_index = exog # used after this `if` block
            exog = model.stoch[exog_index]
        except IndexError:
            n_exog = len(model.stoch)
            raise ValueError(
                f"shock index: {exog}. there are only {n_exog} stochastic vars"
            ) from None

    else:
        exog_index = model.stoch.get_loc(exog)

    # generate one-period shock of size `size` (in standard deviation space)
    n_stochs = len(model.stoch)
    shocks = np.zeros((periods, n_stochs), dtype=float)
    shocks[0, exog_index] = size

    # run simulation with single impulse
    sim = simulate(model, shocks=shocks)
    paths = sim.paths
    impulse = paths - model.ss.values

    response = ImpulseResponse(model, exog=exog, impulse=impulse, size=size)
    return response
