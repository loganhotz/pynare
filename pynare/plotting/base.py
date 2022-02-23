"""
common plots
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt



class PynarePlot(object):

    def __init__(
        self,
        data: np.ndarray,
        labels: Iterable[str] = None
    ):
        self.data = data
        if labels is None:
            self.labels = np.array(['' for _ in range(self._nseries)])
        else:
            self.labels = np.array(labels, dtype=str)


    @property
    def _nseries(self):
        return self.data.shape[1]


    def _plot(self, *args, **kwargs):
        raise NotImplementedError("`_plot` method must be defined by a subclass")


    def _plot_grid(self, **kwargs):

        nrows, ncols = self.axes.shape
        nsubs = self.axes.size

        nplots = min(nsubs, self._nseries)
        for k in range(nplots):
            sub = self.data[:, k::nsubs]
            labels = self.labels[k::nsubs]

            aidx = k % nsubs
            acol, arow = aidx % ncols, aidx // ncols
            ax = self.axes[arow, acol]

            self._plot(ax, sub, labels, **kwargs)


    def _plot_column(self, **kwargs):

        nsubs = self.axes.size

        nplots = min(nsubs, self._nseries)
        for k in range(nplots):
            ts = self.data[:, k]
            label = self.labels[k]

            ax = self.axes[k]
            self._plot(ax, ts, label, **kwargs)


    def arrange_plot(
        self,
        subplots: bool = True,
        layout: Union[str, tuple] = 'rect',
        **kwargs
    ):
        # a lot of the charts in pynare will have very small y-axis limits, which
        #    will run into adjacent plot's areas. tight layout mostly prevents this
        kwargs.setdefault('tight_layout', True)

        if subplots:
            # choose the appropriate arrangement of subplot axes
            if layout:

                if isinstance(layout, str):
                    if layout == 'rect':
                        ncols, nrows = _factor_near_square_root(self._nseries)

                    elif layout == 'square':
                        n_vars = self._nseries

                        # check if n_vars is a square root
                        root = np.sqrt(n_vars)
                        if np.power(int(root + 0.5), 2) == n_vars:
                            nrows, ncols = int(root), int(root)
                        else:
                            root_p1 = int(np.floor(root)) + 1
                            nrows, ncols = root_p1, root_p1

                    elif layout == 'column':
                        nrows, ncols = self._nseries, 1

                    else:
                        raise ValueError(f"unrecognized value of layout: {layout}")

                else:
                    try:
                        nrows, ncols = layout
                    except ValueError:
                        raise ValueError(
                            f"`layout` must be a 2-tuple or a recognized str"
                        ) from None

            else:
                raise ValueError("if `subplots` is True, `layout` must be provided")

            kwargs.setdefault('nrows', nrows)
            kwargs.setdefault('ncols', ncols)

        # create a single Axes object or numpy array of axes, based on `subplots`
        self.figure, self.axes = _generate_subplots(kwargs)

        if subplots:
            if len(self.axes.shape) == 1:
                self._plot_column(**kwargs)
            else:
                self._plot_grid(**kwargs)

        else:
            self._plot(self.axes, self.data, self.labels, **kwargs)

        return self.axes

    def plot(
        self,
        subplots: bool = True,
        layout: Union[str, tuple] = 'rect',
        **kwargs
    ):
        """
        plot data, that may or may not be labeled, on a single axes, a column
        of axes, or a grid of axes. the appearance of the individual subplots
        will vary by the `_plot` method of the subclass

        Parameters
        ----------
        subplots : bool ( = True )
            if False, all plots are drawn on the same subplot. if True, plots
            are drawn on multiple subplots whose arrangement is determined by
            the `layout` parameter
        layout : str | tuple ( = 'rect' )
            if `subplots` is True, this determines the arrangement of the subplots.
            - if `layout = 'rect'`, the subplots are arranged as (nrows, ncols) such
                that nrows*ncols = nseries, and nrows < ncols.
            - if `layout = 'square'`, the subplots are arranged as (n, n) such that
                n*n = nseries and n is chosen to minimize the number of empty plots
            - if `layout = 'column'`, the subplots are arranged in a single column
                with as many rows as there are data series
            - if `layout = 'row'`, the subplots are arranged in a single row with as
                many columns as there are data series
            - if `layout` is a 2-tuple, the tuple determines the arrangement
        **kwargs : keyword arguments
            if a key is included in the documentation of either pyplot.subplots() or
            a pyplot.Figure initialization, that (key, value) pair is passed to
            pyplot.subplots() to create the current Figure and Axes objects. if the
            key is not present in the documentation, it is passed to the subclass's
            `_plot` method

        Returns
        -------
        AxesSubplot | np.ndarray of AxesSubplots
        """
        return self.arrange_plot(subplots, layout, **kwargs)



class PathPlot(PynarePlot):
    """
    plotting column-oriented time series as lines
    """

    def _plot(self, ax, data, label, **kwargs):
        # plot data first to get ylims
        ax.plot(data, **kwargs)

        # if the y-axis displays zero, add a thin x-axis
        ymin, ymax = ax.get_ylim()
        if (ymin < 0) & (ymax > 0):
            ax.axhline(0, linewidth=0.8, color='black', zorder=0)

        if isinstance(label, str):
            ax.legend([label])
        else:
            try:
                if label.size != 0:
                    ax.legend(label)
            except AttributeError:
                import warnings
                warnings.warn("`label` was provided, but was not a str or array")



# arguments for initializing matplotlib subplots and figures
_subplots_kwargs = [
    'nrows',
    'ncols',
    'sharex',
    'sharey',
    'squeeze',
    'subplot_kw',
    'gridspec_kw'
]
_figure_kwargs = [
    # figure method call
    'num',
    'figsize',
    'dpi',
    'facecolor',
    'edgecolor',
    'frameon',
    'FigureClass',
    'clear',
    'tight_layout',
    'constrained_layout',
    # Figure class initialization
    'linewidth',
    'subplotpars'
]
_subplot_figure_kwargs = _subplots_kwargs + _figure_kwargs

def _generate_subplots(kwargs_dict):
    """
    create a set of matplotlib axes from the plt.subplots call
    """
    _kwargs = {}
    for k in list(kwargs_dict.keys()):
        if k in _subplot_figure_kwargs:
            _kwargs[k] = kwargs_dict.pop(k)

    return plt.subplots(**_kwargs)



def _factor_near_square_root(n):
    """
    given an integer `n`, compute the factors closest to sqrt(n). taken from

    https://stackoverflow.com/questions/39248245/factor-an-integer-to-something-
    as-close-to-a-square-as-possible
    """

    v1 = np.ceil(np.sqrt(n))
    v2 = int(n/v1)

    while v2 * v1 != float(n):
        v1 = v1 - 1
        v2 = int(n/v1)

    return int(v1), int(v2)
