"""
utility functions for numpy- and scipy-related operations
"""

from __future__ import annotations

import scipy
import numpy as np



def find_first(
    vec: np.ndarray
):
    # very fast numpy-pure method for finding first index where true
    idx = vec.view(bool).argmax() // vec.itemsize
    return idx if vec[idx] else -1



def hermitian(arr: np.ndarray):
    """
    simple function for computing the hermitian (complex conjugate) of an array
    """
    return np.transpose(np.conjugate(arr))



def matmul_scalar(
    x1, x2,
    out=None, **kwargs
):
    """
    numpy `matmul` does not accept a scalar arguments; this is just a thin wrapper
    for that method. if either `x1` or `x2` are scalar, returns `x1 * x2`. otherwise,
    the procedure follows `matmul`.

    Parameters
    ----------
    x1, x2 : array_like | scalar
        input arrays, scalars are allowed
    out : ndarray ( = None )
        a location where the results are stored
    kwargs : keyword arguments
        keywords to pass to `numpy.matmul` if `x1` or `x2` are not scalars

    Returns
    -------
    ndarray
    """
    try:
        x1_ = x1.item()
    except ValueError:
        x1_ = x1

    try:
        x2_ = x2.item()
    except ValueError:
        x2_ = x2

    if np.isscalar(x1_) or np.isscalar(x2_):
        return x1_ * x2_
    else:
        return np.matmul(x1_, x2_, out=out, **kwargs)


def ensure_2darray(
    arrs: Union[np.ndarray, Iterable[np.ndarray]],
    orient='column'
):
    """
    ensure a single array, or sequence of arrays, has two dimensions. scalars
    are translated to 2-d arrays of size (1, 1), and vectors of length `n` are
    extended to arrays of size (n, 1) or (1, n), depending on the value passed to
    the `orient` parameter

    Parameters
    ----------
    arrs : numpy ndarray | Sequence[numpy ndarray]
        a single numpy ndarray with dimension 0, 1, or 2, or a sequence of arrays
        with those dimensions. if any array passed has dimension > 2, a ValueError
        is thrown
    orient : str ( = 'column' )
        if any array passed through `arrs` is 1-dimensional, this determines whether
        its 2-dimensional representation will have an axis of size 1 as its first or
        second dimension. options are 'column' and 'row'

    Returns
    -------
    numpy ndarray | Sequence[numpy ndarray]
    """

    if orient not in ('column', 'row'):
        o = ('column', 'row')
        raise ValueError(f"acceptable values of `orient` are {', '.join(o)}: {orient}")

    if orient == 'column':
        _extend = lambda x: x[:, np.newaxis]
    else:
        _extend = lambda x: x[np.newaxis, :]

    def _extend_dims(a):
        try:
            sz = len(a.shape)
        except AttributeError:
            if isinstance(a, (int, float, complex)):
                sz = 0
            else:
                raise TypeError("elements must be numpy arrays or scalar numbers")

        if sz == 2:
            return a
        if sz == 1:
            return _extend(a)
        if sz == 0:
            return np.array([[a]])
        else:
            raise ValueError("arrays must be 2-dimensional at most")

    if isinstance(arrs, np.ndarray):
        return _extend_dims(arrs)

    else:
        seq = arrs.__class__
        return seq([_extend_dims(a) for a in arrs])



def concatenate_maybe_1d(
    arrs,
    orient='column',
    *args, **kwargs
):
    """
    concatenate a sequence of arrays (assumed 2-d at most, for now), some elements
    of which may be 1-d arrays, 0-d arrays, or scalars.
        if any of the elements of `arrs` is an array of more than two dimensions,
    a ValueError is thrown

    Parameters
    ----------
    arrs : Sequence[ndarray]
        a sequence of flat numpy arrays to concatenate together.
    orient : str ( = 'column' )
        if any array passed through `arrs` is 1-dimensional, this determines whether
        its 2-dimensional representation will have an axis of size 1 as its first or
        second dimension. options are 'column' and 'row'
    args, kwargs : positional and keyword arguments
        arguments that are passed to np.concatenate(arrs, *args, **kwargs)

    Returns
    -------
    concatenated array
    """

    arrays = ensure_2darray(arrs, orient=orient)
    concat = np.concatenate(arrays, *args, **kwargs)

    return concat.squeeze()



def eigvals_sorted(
    A: np.ndarray,
    order: str = 'descending',
    k: Union[int, Any] = None
):
    """
    return the eigenvalues of a matrix an ascending or descending order. the order
    is determined by magnitude, to handle real and complex entries harmoniously

    Parameters
    ----------
    arr : numpy ndarray
        the square matrix whose eigenvalues are to be computed
    order : str ( = 'descending' )
        determines if eigenvalues should be in ascending or descending order
    k : int | None ( = None )
        the number of sorted eigenvalues to return. if `None` all are returned

    Returns
    -------
    eigvals : np.ndarray
        the sorted eigenvalues
    """
    if order not in ('descending', 'ascending'):
        raise ValueError(f"'order' must be one of 'descending' or 'ascending'")

    eigs = np.linalg.eigvals(A)
    mag_order = np.argsort(np.absolute(eigs))

    if order == 'descending':
        sorted_eigs = eigs[mag_order[::-1]]
    else:
        sorted_eigs = eigs[mag_order]

    if k is None:
        return sorted_eigs
    else:
        return sorted_eigs[:k]



def partitioned_qz(
    A: np.ndarray,
    B: np.ndarray
):
    """
    Given two (n x n) matrices, computes the QZ (aka Generalized Schur)
    decomposition & generalized eigenvectors. Then the resulting right unitary
    matrix and triangular matrices are partitioned according to the magnitude
    of the eigenvalues. The eigenvalues & partitioned matrices are then returned

    More specifically, given the eigenvalue problem A*x = s*B*x, the matrices
    A and B are decomposed as
        A = Q*T*Z
        B = Q*S*Z
    where
        T is upper triangular,
        S is quasi upper triangular,
        Q and Z are unitary matrices

    For matrices X = T, S, Z, they are all partitioned as
        X = [[X_11 X_12]
              X_21 X_22]]
    according to whether the eigenvalues are greater than or less than unity in
    absolute value, with those associated with eigenvalues less than 1 in the
    upper left submatrix

    Parameters
    ----------
    A : numpy 2d-array
        an (n x n) square matrix
    B : numpy 2d-array
        an (n x n) square matrix

    Returns
    -------
    eigs : numpy 1d-array
        n-vector of generalized eigenvalues
    Z_11 : numpy 2d-array
        upper left submatrix of unitary matrix Z
    Z_12 : numpy 2d-array
        upper right submatrix of unitary matrix Z
    Z_21 : numpy 2d-array
        lower left submatrix of unitary matrix Z
    Z_22 : numpy 2d-array
        lower right submatrix of unitary matrix Z
    T_11 : numpy 2d-array
        upper left submatrix of triangular matrix T
    T_12 : numpy 2d-array
        upper right submatrix of triangular matrix T
    T_22 : numpy 2d-array
        lower right submatrix of triangular matrix T
    S_11 : numpy 2d-array
        upper left submatrix of triangular matrix S
    S_12 : numpy 2d-array
        upper right submatrix of triangular matrix S
    S_22 : numpy 2d-array
        lower right submatrix of triangular matrix S
    """

    S, T, alph, bet, Q, Zt = scipy.linalg.ordqz(B, A, sort='iuc')

    with np.errstate(divide='ignore', invalid='ignore'):
        # locally silence RuntimeWarnings about dividing by zero or infinity
        eigs = alph/bet

    # explosive eigenvalues
    expl = find_first((eigs > 1) | np.isinf(eigs))

    # unitary matrix first
    Z = np.transpose(Zt)
    Z_11 = Z[:expl, :expl]
    Z_12 = Z[:expl, expl:]
    Z_21 = Z[expl:, :expl]
    Z_22 = Z[expl:, expl:]

    # triangular matrix associated with matrix A
    T_11 = T[:expl, :expl]
    T_12 = T[:expl, expl:]
    T_22 = T[expl:, expl:]

    # triangular matrix associated with matrix B
    S_11 = S[:expl, :expl]
    S_12 = S[:expl, expl:]
    S_22 = S[expl:, expl:]

    return eigs, Z_11, Z_12, Z_21, Z_22, T_11, T_12, T_22, S_11, S_12, S_22



def array_shingle(
    a0: np.ndarray,
    a1: np.ndarray,
    overlap: int
):
    """
    Merges two arrays along their first dimension by `covering up' the first
    `overlap' rows of the second array with the last `overlap' rows of the
    first array. If the two arrays are 2-dimensional, the analogy to roofing
    shingles is more clearly seen. All dimensions (except the first) of both
    arrays must be equal, or else a ValueError is thrown.

    Parameters
    ----------
    a0 : numpy ndarray
        the array on top that will cover up the bottom array
    a1 : numpy ndarray
        the bottom array that will be covered up by a0
    overlap : int
        the number of rows of the bottom array to be covered up

    Returns
    -------
    shingle : numpy ndarray
        the conjoined array

    Example
    -------
    >>> a = np.random.rand(5, 3)
    [[0.87599626 0.88090669 0.24360141]
      [0.21553097 0.57258704 0.75483751]
      [0.05710367 0.85748782 0.86286493]
      [0.92451993 0.63332986 0.96809019]
      [0.00214015 0.95635431 0.25054558]]

    >>> b = np.random.rand(7, 3)
    [[0.46366265 0.34500481 0.25021755]
      [0.2536837  0.02746545 0.92360626]
      [0.78506297 0.1625407  0.47006583]
      [0.80043353 0.33777573 0.1051361 ]
      [0.84375664 0.48336254 0.41655925]
      [0.88435366 0.76542689 0.77689997]
      [0.13782912 0.7792985  0.17253379]]

    >>> array_shingle(a, b, 2)
    [[0.87599626 0.88090669 0.24360141]
      [0.21553097 0.57258704 0.75483751]
      [0.05710367 0.85748782 0.86286493]
      [0.92451993 0.63332986 0.96809019]
      [0.00214015 0.95635431 0.25054558]
      [0.84375664 0.48336254 0.41655925]
      [0.88435366 0.76542689 0.77689997]
      [0.13782912 0.7792985  0.17253379]]
    """

    s0, s1 = a0.shape, a1.shape

    if s0[1:] != s1[1:]:
        raise ValueError(
            'arrays must be have identical shapes, excluding first dim.'
        )

    shingle = np.zeros((s0[0] + s1[0] - overlap, *s0[1:]))
    shingle[:s0[0], ...] = a0
    shingle[s0[0]:, ...] = a1[overlap:, ...]

    return shingle



def matrix_kronecker_product(A, B, C=None):
    """
    efficiently computes the product A x kron(B, C) where kron(X, Y) is the kronecker
    product of the matrices X and Y and `x` is the standard matrix product. if `C` is
    not provided, A x kron(B, B) is returned

    Parameters
    ----------
    A : numpy ndarray
        the left matrix in usual matrix product
    B : numpy ndarray
        the left matrix in the kronecker product. if `C` is not provided, this is
        used as the right matrix in that product too
    C : numpy ndarray ( = None )
        if provided, this is the right matrix in the kronecker product

    Returns
    -------
    matrix_kronecker : numpy ndarray
    """

    if C is None:
        K = np.kron(B, B)
    else:
        K = np.kron(B, C)

    if (A.shape[1] == 1) and (K.shape[0] == 1):
        return np.outer(A, K)
    else:
        return np.matmul(A, K)



def generalized_sylvester(
    A: np.ndarray,
    B: np.ndarray,
    C: np.ndarray,
    D: np.ndarray,
    E: np.ndarray,
):
    """
    solves the general sylvester equation
                A * X * B^t + C * X * D^t = E        (general sylv)
    where the arrays have sizes
        A, C: (m, m)
        B, D: (n, n)
        X, E: (m, n)

    the implementation follows the process described in `Solution of the Sylvester
    Matrix Equation AXB^t + CXD^t = E` by Gardiner, Laub, Amato, & Moler (1992)
    """

    original_mn = E.shape
    A, B, C, D, E = ensure_2darray((A, B, C, D, E), orient='column')

    # checking shapes for the left arrays
    if A.shape != C.shape:
        a_s, c_s = A.shape, C.shape
        raise ValueError(
            f"`A` and `C` arrays must have the same shape. {a_s} != {c_s}"
        )

    m, mc = A.shape
    if m != mc:
        raise ValueError("`A` and `C` must be square arrays")

    # checking shapes for the right arrays
    if B.shape != D.shape:
        b_s, d_s = B.shape, D.shape
        raise ValueError(
            f"`B` and `D` arrays must have the same shape. {b_s} != {d_s}"
        )

    n, nc = B.shape
    if n != nc:
        raise ValueError("`B` and `D` must be square arrays")

    # final check is on the constant array
    er, ec = E.shape
    if (m != er) or (n != ec):
        raise ValueError(
            "constant array `E` does not conform to sizes of LHS arrays: "
            f"({m}, {n}) != ({er}, {ec})"
        )

    # handling the special case where there is only one row or one column
    if m == 1:
        # B and D are scalars
        aB, cD = A*B, C*D

        X = scipy.linalg.solve(aB + cD, np.transpose(E))
        return np.transpose(X).reshape(original_mn)

    if n == 1:
        # B and D are scalars
        bA, dC = B*A, D*C

        X = scipy.linalg.solve(bA + dC, E)
        return X.reshape(original_mn)

    # algorithm increases in efficiency as (m/n) -> infinity
    if m / n > 1:
        final_transpose = False

    else:
        final_transpose = False
        """
        final_transpose = True

        trans = lambda arrs: tuple([np.transpose(a) for a in arrs])
        B, A, D, C, E = trans((A, B, C, D, E))

        # switch indices after transposition
        m, n = E.shape
        """

    # steps 1 and 2 are to find generalized schur decompositions of the pairs
    #    of left and right matrices. for the first step, we have that
    #        A = Q1 * P * Z1^t    and        C = Q1 * S * Z1^t
    #    where `*` is matrix multiplication and `^t` denotes the transpose. Q1 and
    #    Z1 are unitary; P is quasi-upper-triangular, with blocks of size one or two,
    #    depending on whether or not the k-th eigval is complex. S is upper triangular
    P, S, Q1, Z1 = scipy.linalg.qz(A, C, output='real')
    T, R, Q2, Z2 = scipy.linalg.qz(D, B, output='real')

    # step three left-multiplies both sides of Equation (general sylv) by Q1^T and
    #    right-multiplies by Q2
    F = np.matmul(np.transpose(Q1), np.matmul(E, Q2))

    # pre-allocate rotated output matrix
    Y = np.full(F.shape, np.nan, dtype=float)

    # step four is described at the end of section 2 of Gardiner et. al.
    #    essentially a modified bartels-stewart method.
    k = n - 1
    for i in range((T.diagonal(offset=-1) == 0).sum()+1):
        rP = R[k, k] * P
        tS = T[k, k] * S

        off_diag = T[k, k-1]
        if (off_diag != 0) and (k != 0):
            # eigenvalue block is (2, 2). solve two columns at once

            # remove influence of already-computed Y columns
            ykm1, yk = np.zeros(m, dtype=float), np.zeros(m, dtype=float)
            for j in range(n-1-k):
                ykm1 = ykm1 + np.dot(R[k-1, k+1+j]*P + T[k-1, k+1+j]*S, Y[:, k+1+j])
                yk = yk + np.dot(R[k, k+1+j]*P + T[k, k+1+j]*S, Y[:, k+1+j])

            f = np.hstack((F[:, k-1] - ykm1, F[:, k] - yk))

            # construct (2m, 2m) linear system that jointly determines columns k
            #    and k-1 of Y
            northwest = R[k-1, k-1]*P + T[k-1, k-1]*S
            northeast = R[k-1, k]*P + T[k-1, k]*S
            southwest = T[k, k-1] * S
            southeast = rP + tS
            block = np.block([[northwest, northeast], [southwest, southeast]])

            # solve; partition
            Y_stack = scipy.linalg.solve(block, f)
            Y[:, k-1] = Y_stack[:m]
            Y[:, k] = Y_stack[m:]

            k = k - 2

        else:
            # eigenvalue block is (1, 1). solve for one column of Y

            # remove influence of already-computed Y columns
            yk = np.zeros(m, dtype=float)
            for j in range(n-1-k):
                yk = yk + np.dot(R[k, k+1+j]*P + T[k, k+1+j]*S, Y[:, k+1+j])

            Y[:, k] = scipy.linalg.solve(rP+tS, F[:, k] - yk)

            k = k - 1

    # rotate output matrix and transpose if m < n
    X = np.matmul(Z1, np.matmul(Y, np.transpose(Z2)))
    if final_transpose:
        return np.transpose(X)
    return X



def inv_sym_pd(arr: np.ndarray):
    """
    quickly invert a symmetric positive definite matrix. curiously, scipy and numpy
    don't have a built-in method for this already, although scipy.linalg does have
    symmetric flags for its `solve` method

    uses Kerrick Staley's solution in this answer:
        https://stackoverflow.com/questions/40703042/more-efficient-way-to-invert
        -a-matrix-knowing-it-is-symmetric-and-positive-semi

    Parameters
    ----------
    arr : np.ndarray
        square symmetric matrix

    Returns
    -------
    inv : np.ndarray
        inverse of `arr`
    """
    cholesky, info = scipy.linalg.lapack.dpotrf(arr)
    if info != 0:
        raise ValueError("could not perform Cholesky decomposition")

    inv, info = scipy.linalg.lapack.dpotri(cholesky)
    if info != 0:
        raise ValueError("could not invert Cholesky triangular array")

    upper_triangular_to_symmetric(inv)
    return inv



triangular_cache = {}
def upper_triangular_to_symmetric(ut):
    """
    used in `inv_sym_pd` to quickly convert in-place an upper-triangular matrix to
    its corresponding symmetric analogue
    """
    n = ut.shape[0]
    try:
        idx = triangular_cache[n]
    except KeyError:
        idx = np.tri(n, k=-1, dtype=np.bool)
        triangular_cache[n] = idx

    ut[idx] = ut.T[idx]
