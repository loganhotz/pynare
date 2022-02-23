"""
create Summary objects for Models, Simulations, Solutions, etc.
"""
from __future__ import annotations


import numpy as np
from itertools import product
from functools import cached_property

import pynare.utils.numpy as unp



class Summary(object):

    features = ()

    def __init__(self, obj, sig: int = 6, pad: int = 4):
        self.obj = obj
        self.sig = sig
        self.pad = ' '*pad

    def __repr__(self):
        class_ = self.__class__.__name__
        return class_

    __str__ = __repr__



class ModelSummary(Summary):

    features = (
        'variables',
        'eigen',
        'stoch'
    )

    def __init__(
        self,
        model: Model,
        sig: int = 6,
        pad: int = 4,
        variables: bool = True,
        eigen: bool = True,
        stoch: bool = True
    ):
        super().__init__(model, sig, pad)

        self.variables = variables
        self.eigen = eigen
        self.stoch = stoch

    def variables_str(self):
        """
        summary of variables in the model.
        """
        labels = (
            'variables', 'stochastic shocks', 'state vars',
            'forward-looking vars', 'static vars'
        )
        lpad = max(map(len, labels))

        variables = (
            self.obj.endog,
            self.obj.stoch,
            self.obj.state_vars,
            self.obj.forward_vars,
            self.obj.static_vars
        )
        var_strs = ['variables:']
        for l, v in zip(labels, variables):
            nl, nv = len(l), len(v)
            if nv > 0:
                vstr = f"number of {l}: {' '*(lpad-nl)}{nv}"
                var_strs.append(vstr)

        delim = f"\n{self.pad}"
        var_block = delim.join(var_strs)
        return var_block

    def eigen_str(self):
        """
        format real parts, complex parts, and modulus of state-space eigenvalues
        """

        # sort eigenvalues in ascending modulus order
        unsorted = self.obj.statespace.eigen
        mod_order = np.argsort(np.absolute(unsorted))

        # partition parts to be displayed
        eigs = unsorted[mod_order]
        r, c, m = np.real(eigs), np.imag(eigs), np.absolute(eigs)

        # record number of points to left and right of decimal
        arr = np.absolute(np.concatenate((r, c, m)))
        good = (arr > 0) & np.isfinite(arr) & ~np.isnan(arr)
        with np.errstate(divide='ignore', invalid='ignore'):
            # locally silence RuntimeWarnings about dividing by zero or infinity
            l = int(np.floor(np.max(np.log10(arr[good]))) + 3)
        d = self.sig

        # float fmt depends on significant digit and magnitude of largest float
        fmt = lambda x: f"{x:{l+d}.{d}f}"

        headings = ('real', 'imag', 'mod')
        heading = ' '.join([h.rjust(l+d) for h in headings])

        eig_strs = ['steady-state eigenvalues:', heading]
        for r_, c_, m_ in zip(r, c, m):
            estr = f"{fmt(r_)} {fmt(c_)} {fmt(m_)}"
            eig_strs.append(estr)

        delim = f"\n{self.pad}"
        eig_block = delim.join(eig_strs)
        return eig_block

    def stoch_str(self):
        """
        print the covariance matrix of shocks
        """
        stoch = self.obj.stoch.names
        lpad = max(map(len, stoch)) + 1

        # record number of points to left and right of decimal
        with np.errstate(divide='ignore', invalid='ignore'):
            # locally silence RuntimeWarnings about dividing by zero or infinity
            l = int(np.floor(np.nanmax(np.log10(np.absolute(self.obj.sigma)))) + 3)
        d = self.sig

        # float fmt depends on significant digit and magnitude of largest float
        fmt = lambda x: f"{x:{l+d}.{d}f}"

        # header row of string, then column labels
        cols = ' '*lpad + ' '.join([st.rjust(l+d) for st in stoch])
        stoch_strs = ['covariance of stoch shocks:', cols]

        for st, row in zip(stoch, self.obj.sigma):
            sstr = st.ljust(lpad) + ' '.join(map(fmt, row))
            stoch_strs.append(sstr)

        delim = f"\n{self.pad}"
        stoch_block = delim.join(stoch_strs)
        return stoch_block

    def __str__(self):
        heading = 'model summary:'
        feat = [heading]
        for f in self.features:
            if getattr(self, f):
                f_method = f"{f}_str"
                feat.append(getattr(self, f_method)())
        return '\n\n'.join(feat)



class ModelStatisticsSummary(Summary):

    features = (
        'moments',
        'corr',
        'autocorr',
        'decomp'
    )

    def __init__(
        self,
        model: Model,
        sig: int = 6,
        pad: int = 4,
        moments: bool = True,
        corr: bool = True,
        autocorr: bool = 5,
        decomp: bool = True
    ):
        super().__init__(model, sig, pad)

        self.moments = moments
        self.corr = corr
        self.autocorr = autocorr
        self.decomp = decomp

    def moments_str(self):
        """
        displays the mean, standard deviation, and variance of endogenous variables
        """

        # use the ModelStatistics' classes methods
        mean = self.obj.mean()
        std_dev = self.obj.std()
        var = np.square(std_dev)

        # column labels are moment names
        headings = ('mean', 'std dev', 'var')
        arr = np.transpose(np.vstack((mean, std_dev, var)))

        arr_str = _summarize_array(
            arr=arr,
            labels=headings,
            sig=self.sig,
            use_labels=True
        )

        # row labels for each endogenous variable names
        endog = [''] + list(self.obj.model.endog.names)
        lpad = max(map(len, endog)) + 1
        labels = [en.ljust(lpad) for en in endog]

        # add heading and pad all rows
        mom_str = ['model moments:'] + [''.join((l, a)) for l, a in zip(labels, arr_str)]
        delim = f"\n{self.pad}"
        return delim.join(mom_str)

    def corr_str(self):
        """
        displays the correlation matrix of the endogenous variables
        """
        # use the ModelStatistics' class methods
        corr = self.obj.corr()

        # column labels to display
        endog = self.obj.model.endog.names
        arr_str = _summarize_array(
            arr=corr,
            labels=endog,
            sig=self.sig,
            use_labels=True
        )

        # add an empty row and left-align the endog names
        lpad = max(map(len, endog)) + 1
        labels = [' '*lpad] + [en.ljust(lpad) for en in endog]

        # add heading and pad all the rows
        cor_str = ['correlation:'] + [''.join((l, a)) for l, a in zip(labels, arr_str)]
        delim = f"\n{self.pad}"
        return delim.join(cor_str)

    def autocorr_str(self):
        """
        displays the autocorrelation structure of the endogenous variables
        """
        # use the ModelStatistics' class methods
        acr_3d = self.obj.autocorr(p=self.autocorr)
        arr = np.transpose(np.array([np.diagonal(arr) for arr in acr_3d]))

        # column labels to display
        headings = [f"p = {i}" for i in range(self.autocorr)]
        arr_str = _summarize_array(
            arr=arr,
            labels=headings,
            sig=self.sig,
            use_labels=True
        )

        # row labels for each endogenous variable names
        endog = [''] + list(self.obj.model.endog.names)
        lpad = max(map(len, endog)) + 1
        labels = [en.ljust(lpad) for en in endog]

        # add heading and pad all rows
        acr_str = ['autocovariance of endogenous variables:']
        acr_str = acr_str + [''.join((l, a)) for l, a in zip(labels, arr_str)]
        delim = f"\n{self.pad}"
        return delim.join(acr_str)

    def decomp_str(self):
        """
        displays the variance decomposition of the endogenous variables
        """

        # use the ModelStatistics' classes methods
        decomp = self.obj.var_decomp()

        # column labels to display
        headings = self.obj.model.stoch.names
        arr_str = _summarize_array(
            arr=decomp,
            labels=headings,
            sig=self.sig,
            use_labels=True
        )

        # row labels for each endogenous variable names
        endog = [''] + list(self.obj.model.endog.names)
        lpad = max(map(len, endog)) + 1
        labels = [en.ljust(lpad) for en in endog]

        # add heading and pad all rows
        decomp_str = ['variance decomposition:']
        decomp_str = decomp_str + [''.join((l, a)) for l, a in zip(labels, arr_str)]
        delim = f"\n{self.pad}"
        return delim.join(decomp_str)

    def __str__(self):
        heading = 'statistics summary:'
        feat = [heading]
        for f in self.features:
            if getattr(self, f):
                f_method = f"{f}_str"
                feat.append(getattr(self, f_method)())
        return '\n\n'.join(feat)



class SolutionSummary(Summary):

    def __init__(self, model: Model, sig: int, pad: int, thresh: int, order: int):
        super().__init__(model, sig, pad)
        self.thresh = thresh
        self.order = order



class FirstOrderSummary(SolutionSummary):

    def __init__(
        self,
        sol: ModelSolution,
        sig: int = 6,
        pad: int = 4,
        thresh: int = None
    ):
        super().__init__(sol, sig, pad, thresh, 1)

    def policy_arrays(self):

        # prepare column labels - the endogenous variable names
        endog = self.obj.model.endog.names
        idx = self.obj.model.indexes

        # steady-state values are already in declaration order
        ss_dr = self.obj.model.ss.values[idx.dr_order]

        # join arrays so that both gy and gu use same widest value
        ss, gy, gu = unp.ensure_2darray((ss_dr, self.obj.gy, self.obj.gu))

        # reorder gy columns (displayed as rows in string) to declaration order
        #   to align with labels, and match stoch vars which are also in dc order
        dr_idx = idx.dr_order
        gy = gy[:, np.argsort(dr_idx[np.in1d(dr_idx, idx.state)])]

        arr = np.transpose(np.hstack((ss, gy, gu)))

        arr_dc = arr[:, self.obj.model.indexes.dc_order]
        arr_str = _summarize_array(
            arr=arr_dc,
            labels=endog,
            sig=self.sig,
            use_labels=True,
            thresh=self.thresh
        )

        # with widths matched, we split the arrays
        n_gy, n_gu = gy.shape[1], gu.shape[1]
        gy_str, gu_str = arr_str[:n_gy+2], arr_str[-n_gu:]
        return gy_str, gu_str

    def row_labels(self):

        state = [f"{st}(-1)" for st in self.obj.model.state_vars.names]
        stoch = self.obj.model.stoch.names

        # join variable names and find maximum length
        labels = np.concatenate((['', 'steady'], state, [''], stoch))
        lpad = max(map(len, labels)) + 1

        return [lab.ljust(lpad) for lab in labels]

    def __str__(self):
        heading = 'first order policy and transition functions:'
        feat = [heading]

        # lists of strings for the policy arrays, then state and stoch vars
        gyn, gun = self.policy_arrays()
        labels = self.row_labels()

        # mostly blank row with plus sign in the middle
        n_chars = len(gyn[0])
        plus_sign = ' '*(n_chars // 2) + '+' + ' '*(n_chars // 2 - 1)
        fillers = [plus_sign.rjust(n_chars)]

        policy = gyn + fillers + gun
        for lab, pol in zip(labels, policy):
            row = ''.join((lab, pol))
            feat.append(row)

        delim = f"\n{self.pad}"
        return delim.join(feat)



class SecondOrderSummary(SolutionSummary):

    def __init__(
        self,
        sol: ModelSolution,
        sig: int = 6,
        pad: int = 4,
        thresh: int = None
    ):
        super().__init__(sol, sig, pad, thresh, 2)

        # self.state = self.obj.model.state_vars.names


    @cached_property
    def _threshold_locs(self):

        gy, gu, gyx, gyu, guu, del2 = self._policy_arrays
        arr = np.transpose(np.hstack((gy, gu, gyx, gyu, guu, del2)))
        arr_dc = arr[:, self.obj.model.indexes.dc_order]

        # based on the chosen significant digits and theshold value, select a subset
        #   of the columns to print
        if self.thresh is not None:
            thresh = 10**(-(self.sig + self.thresh))
        else:
            thresh = 10**(-self.sig)
        tloc = ~np.all(arr_dc < thresh, axis=1)

        # always display steady state and del2 array
        tloc[0] = True
        tloc[-1] = True

        return tloc


    @cached_property
    def _policy_locs(self):
        """
        n x 6 array where n is number of rows in policy arrays before removing values
        under threshold
        """
        gy, gu, gyx, gyu, guu, del2 = self._policy_arrays
        arr = np.transpose(np.hstack((gy, gu, gyx, gyu, guu, del2)))
        arr_dc = arr[:, self.obj.model.indexes.dc_order]

        # record number of rows for each array, adjust first one for steady row
        blocks = [arr.shape[1] for arr in (gy, gu, gyx, gyu, guu, del2)]

        # add a zero so indexing into cumulative summed works right
        blocks.insert(0, 0)
        indices = np.cumsum(blocks)

        pol = np.zeros((arr_dc.shape[0], len(indices)-1), dtype=bool)
        for i, end in enumerate(indices[1:]):
            start = indices[i]
            pol[start:end, i] = True

        return pol


    @cached_property
    def _policy_arrays(self):

        # steady-state values are already in declaration order
        ss_dr = self.obj.model.ss.values[self.obj.model.indexes.dr_order]

        # first-order solution arrays
        lower = self.obj.lower_solution
        ss_dr, gy, gu = unp.ensure_2darray((ss_dr, lower.gy, lower.gu))

        # second order solution arrays
        ar = self.obj.arrays
        gyx, gyu, guu, del2 = unp.ensure_2darray((ar.ghxx, ar.ghxu, ar.ghuu, ar.del2))

        # reorder gy columns (displayed as rows in string) to declaration order
        #   to align with labels, and match stoch vars which are also in dc order
        idx = self.obj.model.indexes
        dr_idx = idx.dr_order
        st_idx = np.argsort(dr_idx[np.in1d(dr_idx, idx.state)])

        # reorder the second-order matrices in the same was as gy
        n_st = len(st_idx)
        gyx_idx = n_st*np.repeat(st_idx, n_st) + np.tile(st_idx, n_st)
        gyu_idx = n_st*np.repeat(st_idx, n_st) + np.tile(np.arange(n_st), n_st)

        gy = gy[:, st_idx]
        gyx = gyx[:, gyx_idx]
        gyu = gyu[:, gyu_idx]

        gy = np.hstack((ss_dr, gy))
        return gy, gu, gyx, gyu, guu, del2


    def row_labels(self):

        state = [f"{st}(-1)" for st in self.obj.model.state_vars.names]
        stoch = [sc for sc in self.obj.model.stoch.names]

        first_order = ['steady'] + state + stoch

        # prepare second-order labels
        state2 = [', '.join(s2) for s2 in product(state, state)]
        stoch2 = [', '.join(s2) for s2 in product(stoch, stoch)]
        state_stoch = [', '.join(s2) for s2 in product(state, stoch)]

        # join and select those that meet threshold
        all_labels = np.array(first_order + state2 + state_stoch + stoch2 + ['del2'])
        labels = all_labels[self._threshold_locs]

        # find maximum length and left-pad
        lpad = max(map(len, labels)) + 1
        labels = np.array(list(map(lambda x: x.ljust(lpad), labels)))

        # partition into six lists according to the policy block, and return
        ploc = self._policy_locs[self._threshold_locs]
        return tuple([list(labels[ploc[:, i]]) for i in range(ploc.shape[1])])


    def policy_arrays(self):

        # prepare column labels - the endogenous variable names
        endog = self.obj.model.endog.names

        gy, gu, gyx, gyu, guu, del2 = self._policy_arrays
        arr = np.transpose(np.hstack((gy, gu, gyx, gyu, guu, del2)))
        arr_dc = arr[:, self.obj.model.indexes.dc_order]

        # select rows which meet threshold, then cast to array to boolean index
        _arr_str = _summarize_array(
            arr=arr_dc[self._threshold_locs],
            labels=endog,
            sig=self.sig,
            use_labels=True
        )
        endog_str = _arr_str[0]
        arr_str = np.array(_arr_str[1:], dtype=str)

        # partition into six arrays and return
        ploc = self._policy_locs[self._threshold_locs]
        return tuple([arr_str[ploc[:, i]] for i in range(ploc.shape[1])]), endog_str


    def __str__(self):
        heading = 'second order policy and transition functions:'
        feat = [heading]

        # lists of strings for the policy arrays, then state and stoch vars
        policy, endog = self.policy_arrays()
        labels = self.row_labels()

        # mostly blank row with plus sign in the middle
        lwidth, pwidth = len(labels[0][0]), len(policy[0][0])
        n_chars = lwidth + pwidth
        plus_sign = ' '*(n_chars // 2) + '+' + ' '*(n_chars // 2 - 1)

        feat.append(endog.rjust(n_chars))
        for i, (lab, pol_arr) in enumerate(zip(labels, policy)):

            if pol_arr.size:
                # some arrays will have no elements that meat threshold. don't print
                for l, p in zip(lab, pol_arr):
                    row = ''.join((l, p))
                    feat.append(row)

                if i < len(labels) - 1:
                    feat.append(plus_sign)

        delim = f"\n{self.pad}"
        return delim.join(feat)



def _summarize_array(
    arr: np.ndarray,
    labels: Iterable[str],
    sig: int,
    use_labels: bool = True,
    thresh: float = None
):
    """

    """

    # optionally select a subset of displayed array
    if thresh is not None:
        thresh_ = 10 ** (-(sig + thresh))
        pidx = ~np.all(arr < thresh_, axis=1)
    else:
        pidx = np.ones(arr.shape[1], dtype=bool)

    # ensure that `labels` is an array so indexing with list works right
    labels = np.array(labels)

    # only select rows that meet threshold
    arr, labels = arr[:, pidx], labels[pidx]

    # number of points to right and left of decimal. `+3` factors in minus signs
    with np.errstate(divide='ignore', invalid='ignore'):
        # locally silence RuntimeWarnings about dividing by zero or infinity
        l = int(np.fix(np.nanmax(np.log10(np.absolute(arr)))) + 3)

    # horizontal spacing determined by width of label variable name or float repr
    w = max([l+sig] + [len(lab) for lab in labels]) + 1

    # float fmt depends on significant digit and magnitude of largest float
    fmt = lambda x: f"{x:{w}.{sig}f}"

    cols = ' '.join([lab.rjust(w) for lab in labels])
    if use_labels:
        arr_strs = [cols]
    else:
        arr_strs = []

    for row in arr:
        try:
            rstr = ' '.join(map(fmt, row))
        except TypeError:
            # tried iterating over a float
            rstr = fmt(row)
        arr_strs.append(rstr)

    return arr_strs
