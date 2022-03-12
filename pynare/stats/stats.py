"""
interface for a model's statistical properties
"""
from __future__ import annotations

from functools import partial

from pynare.io.summary import ModelStatisticsSummary

from pynare.stats.moments import (
    mean,
    var,
    std,
    corr,
    cov,
    autocorr,
    autocov
)

from pynare.stats.var import var_decomp



class ModelStatistics(object):

    features = {
        'mean': mean,
        'var': var,
        'std': std,
        'cov': cov,
        'corr': corr,
        'autocorr': autocorr,
        'autocov': autocov,
        'var_decomp': var_decomp
    }


    def __init__(self, model: Model):
        self.model = model


    def __getattr__(self, attr: str):
        try:
            func = self.features[attr]

            feature_func = partial(func, md=self.model)
            feature_func.__doc__ = func.__doc__
            return feature_func

        except KeyError:
            raise AttributeError(
                f"'ModelStatistics' object has no attribute '{attr}'"
            ) from None


    def summary(self, **kwargs):
        """
        returns a summary of the statistical features of a model with options to
        show the moments, autocorrelation, and variance decomposition of the endogenous
        variables of a model

        Parameters
        ----------
        sig : int ( = 6 )
            the number of digits to display when printing floats
        pad : int ( = 4 )
            the number of spaces to use on the LHS of displayed attributes
        moments, corr, decomp : bool ( = True )
            indicators of features to display
        autocorr : int ( = 5 )
            the number of periods to compute autocorrelations for

        Returns
        -------
        ModelStatisticsSummary
        """

        return ModelStatisticsSummary(self, **kwargs)


    def __repr__(self):
        return 'ModelStatistics'
