
from __future__ import annotations

from functools import cached_property

from pynare.core.variables import ModelCalibration
from pynare.core.generic import DynamicModel
from pynare.core.steady import SteadyState
from pynare.core.solve import (
    FirstOrderSolution,
    SecondOrderSolution,
    OccBinSolution,
    default_order
)

from pynare.stats.stats import ModelStatistics
from pynare.simul import simulate
from pynare.impulse import impulse_response

from pynare.parsing import read_model

from pynare.io.summary import ModelSummary
from pynare.io.files import (
    read_file,
    read_example
)



class Model(DynamicModel):

    def __init__(
        self,
        endog: list = [],
        stoch: list = [],
        determ: list = [],
        params: dict = {},
        shocks: dict = {},
        exprs: list = [],
        local_params: dict = {},
        initial: dict = {},
        terminal: dict = {},
        historical: dict = {},
        language: str = 'dynare',
        name: str = ''
    ):

        self.local_params = ModelCalibration(local_params)

        super().__init__(
            endog=endog,
            stoch=stoch,
            determ=determ,
            params=params,
            shocks=shocks,
            exprs=exprs,
            initial=initial,
            terminal=terminal,
            historical=historical,
            language=language,
            name=name
        )


    @classmethod
    def from_path(
        cls,
        filepath: Union[str, Path],
        language: str = 'dynare'
    ) -> Model:
        """
        create a model from a path name. if file is not found, assume it's a pre-made
        example model in pynare/examples before throwing a `FileNotFoundError`

        Parameters
        ----------
        filepath : str | Path
            path to .mod or .txt file that defines the model
        language : str
            the language the model is written in

        Returns
        -------
        Model
        """
        try:
            model_definition = read_file(filepath)
        except FileNotFoundError:
            try:
                model_definition = read_example(filepath)
            except ValueError:
                raise FileNotFoundError(filepath) from None

        # create the model name from the filepath
        name = filepath.split('/')[-1]
        name_no_suffix = name.split('.')[0]

        # create outline and use that to define model
        model_outline = read_model(model_definition, language)
        return cls.from_outline(model_outline, language, name_no_suffix)


    @classmethod
    def from_outline(
        cls,
        outline: ModelOutline,
        language: str = 'dynare',
        name: str = ''
    ) -> Model:
        """

        """
        # endogenous variables
        endogenous = outline._endogenous

        # declared exogenous variables (just the names)
        exog_stoch = outline._stochastic_exogenous
        exog_determ = outline._deterministic_exogenous

        # parameters and locally declared variables are dicts like {name: value}
        params = outline._parameters
        local_model_vars = outline._local_model_variables

        # abstract syntax trees of the model expressions. steady-state and dynamic
        #    representations are created later
        exprs = outline._model_expression_asts

        # list of shock definitions that set variance, deterministic periods, etc
        shocks = outline._shocks

        # boundary values for endogenous variables
        initial = outline._initial_values
        terminal = outline._terminal_values
        historical = outline._historical_values

        return Model(
            endog=endogenous,
            stoch=exog_stoch,
            determ=exog_determ,
            params=params,
            shocks=shocks,
            exprs=exprs,
            local_params=local_model_vars,
            initial=initial,
            terminal=terminal,
            historical=historical,
            language=language,
            name=name
        )


    def solve(
        self,
        order: int = 0,
        constraint: Sequence[str | Sequence[str]] = [],
        *args, **kwargs
    ):
        """
        compute the solution of the model. produces the linear approximation of the
        solution by default, but can also compute the occbin solution as outlined in
        "OccBin: A Toolkit for Solving Dynamic Models with Occasionally Binding
        Constraints Easily" (2014) by Guerrieri & Iacoviello.

        Parameters
        ----------
        order : int ( = 0 )
            the order at which to compute the linear approximation to the solution
        constraint : Sequence[str | Sequence[str]] ( = [] )
            occasionally-binding constraints to enforce

        Returns
        -------
        ModelSolution
        """
        if constraint:
            # parameters defining how model-switching occurs
            exprs, sigma = kwargs.pop('exprs', {}), kwargs.pop('sigma', {})

            # occasionally-bound solutions are always conditional on shock paths
            exog, shocks = kwargs.pop('exog', None), kwargs.pop('shocks', 100)
            init = kwargs.get('init', None)

            solution = OccBinSolution(
                self,
                constraint=constraint,
                exprs=exprs,
                sigma=sigma,
                exog=exog,
                shocks=shocks,
                init=init
            )

        else:
            if not order:
                order = default_order

            if order == 1:
                solution = FirstOrderSolution(self)
            elif order == 2:
                solution = SecondOrderSolution(self)
            else:
                raise ValueError(f"unrecognized `order` value: {order}")

        self._solution = solution.solve(*args, **kwargs)
        return self._solution


    def simulate(
        self,
        shocks: np.ndarray = None,
        periods: int = 20,
        init: np.ndarray = None,
        order: int = 0
    ):
        """
        simulate a model

        Parameters
        ----------
        shocks : numpy ndarray ( = None )
            the paths of innovations to all the exogenous variables to use in the
            simulation, in percent deviation space. if not provided, the shock paths
            are random normal
        periods : int ( = 20 )
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
        return simulate(self, shocks, periods, init, order)


    def impulse_response(
        self,
        exog: str | Sequence[str] = None,
        periods: int = 20,
        size: int | float = 1
    ):
        """
        compute the impulse response of a model to an exogenous shock

        Parameters
        ----------
        exog : str | Iterable[str] ( = None )
            the exogenous variable to shock. if 'None', the first stochastic variable
            is the one whose impulse response is computed
        periods : int ( = 20 )
            the number of periods to calculate the responses for
        size : int | float ( = 1 )
            the size of the shock (in standard deviation space)

        Returns
        -------
        ImpulseResponse
        """
        return impulse_response(self, exog, periods, size)


    def summary(self, **kwargs):
        """
        returns a summary of the model that (optionally) describes the typology of
        variables, state-space eigenvalues, and covariance of shocks

        Parameters
        ----------
        sig : int = 6
            the number of digits to display when printing eigens and covariance matrices
        pad : int = 4
            the number of spaces to use on the LHS of displayed attributes
        variables, eigen, stoch : bool = True
            indicators of feature display

        Returns
        -------
        ModelSummary
        """
        return ModelSummary(self, **kwargs)


    @property
    def model_scope(self):
        return {**self.params, **self.local_params}


    @property
    def stats(self):
        """
        returns an interface for statistics of the model
        """
        return ModelStatistics(self)


    @property
    def solution(self):
        """
        access the solution of the model. if the class method `solve` has not
        explicitly been called, then the default parameter values are used to
        derive the solution
        """
        try:
            return self._solution
        except AttributeError:
            _ = self.solve(order=default_order)
            return self._solution


    @property
    def solution_order(self):
        """
        returns the solution order. if the model hasn't been solved yet, return None
        """
        try:
            return self._solution.order
        except AttributeError:
            return None


    @property
    def first_order_solution(self):
        """
        returns the first-order solution of the model. if the model is already
        solved at order 1, return. if the model is already solved at order 2, return
        the reference to `lower_solution`. if not solve, solve at default order
        """
        try:
            sol = self._solution
        except AttributeError:
            sol = self.solve(order=default_order)

        if sol.order == 1:
            return sol
        elif sol.order == 2:
            return sol.lower_solution
        else:
            raise NotImplementedError(f"order = {sol.order}")
