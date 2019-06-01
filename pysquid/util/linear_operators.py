import numpy as np
from scipy.ndimage import label
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import LinearOperator


def curl(g, dx=1.0, dy=1.0):
    """
    Given g-field, compute corresponding current field

    input:
        g : float array of shape (Ly, Lx)
    output:
        jx, jy: x and y-currents, float arrays of shape
                (Ly+1,Lx) and (Ly, Lx+1) respectively
    """
    Dh, Dv = makeD(g.shape, dx, dy)
    jx, jy = -Dv.dot(g.ravel()), Dh.dot(g.ravel())
    return jx.reshape(*g.shape), jy.reshape(*g.shape)


def makeD(shape, dx=1.0, dy=1.0):
    """
    Given an image shape, returns sparse matrices
    representing finite-difference derivatives in the horizontal
    and vertical directions
    input:
        shape : tuple of ints (Ly, Lx)
    output:
        dh, dv : scipy sparse COO matrices for computing
                 finite difference derivatives in horiz/vert direction
    """
    Ly, Lx = shape
    assert Ly > 1, "Ly must be greater than 1"
    assert Lx > 1, "Lx must be greater than 1"
    N = Ly * Lx
    tdx, tdy = 2 * dx, 2 * dy
    hrow, hcol, hdata = [], [], []
    vrow, vcol, vdata = [], [], []
    for i in range(N):
        x, y = i % Lx, i // Lx
        hrow += [i, i]
        vrow += [i, i]
        if x > 0 and x < Lx - 1:
            hcol += [i - 1, i + 1]
            hdata += [-1.0 / tdx, 1.0 / tdx]
        else:  # Don't want to compare to zeros outside
            if x == 0:
                hcol += [i, i + 1]
            elif x == Lx - 1:
                hcol += [i - 1, i]
            hdata += [-1.0 / dx, 1.0 / dx]
        if y > 0 and y < Ly - 1:
            vcol += [i - Lx, i + Lx]
            vdata += [-1.0 / tdy, 1.0 / tdy]
        else:
            if y == 0:
                vcol += [i, i + Lx]
            elif y == Ly - 1:
                vcol += [i - Lx, i]
            vdata += [-1.0 / dy, 1.0 / dy]
    dh = csr_matrix((hdata, (hrow, hcol)), shape=(N, N))
    dv = csr_matrix((vdata, (vrow, vcol)), shape=(N, N))
    return dh, dv


def makeD2(shape, dx=1.0, dy=1.0):
    """
    Given an image shape, returns sparse matrices
    representing second derivatives in the horizontal
    and vertical directions
    input:
        shape : tuple of ints (Ly, Lx)
    output:
        d2h, d2v : scipy sparse COO matrices for computing
                   second derivatives in horiz/vert direction
    """
    Ly, Lx = shape
    assert Ly > 2, "Ly must be greater than 2"
    assert Lx > 2, "Lx must be greater than 2"
    N = Ly * Lx
    d2x, d2y = dx * dx, dy * dy
    hrow, hcol, hdata = [], [], []
    vrow, vcol, vdata = [], [], []
    for i in range(N):
        x, y = i % Lx, i // Lx
        hrow += [i, i, i]
        vrow += [i, i, i]
        if x > 0 and x < Lx - 1:
            hcol += [i - 1, i, i + 1]
        else:  # Don't want to compare to zeros outside
            if x == 0:
                hcol += [i, i + 1, i + 2]
            elif x == Lx - 1:
                hcol += [i, i - 1, i - 2]
        hdata += [1.0 / d2x, -2.0 / d2x, 1.0 / d2x]
        if y > 0 and y < Ly - 1:
            vcol += [i - Lx, i, i + Lx]
        else:
            if y == 0:
                vcol += [i, i + Lx, i + 2 * Lx]
            elif y == Ly - 1:
                vcol += [i, i - Lx, i - 2 * Lx]
        vdata += [1.0 / d2y, -2.0 / d2y, 1.0 / d2y]
    d2h = csr_matrix((hdata, (hrow, hcol)), shape=(N, N))
    d2v = csr_matrix((vdata, (vrow, vcol)), shape=(N, N))
    return d2h, d2v

def finite_support(mask, out_shape=None):
    r"""
    Creates finite support operator F, such that
    g = F \tilde{g}, where \tilde{g} has fewer degrees of freedom as
    determined by the mask, which has 1 where g-values should be constant
    and 0 where g-values can be free.

    Parameters
    ----------
    mask : array_like
        Mask of shape (Ly, Lx), with regions of 1 where currents should
        NOT flow, and 0 where they should flow.
    out_shape : tuple
        Required mask shape. If provided, the function only checks that 
        mask.shape = out_shape
    Returns
    -------
    F : scipy.sparse.csr.csr_matrix
        operator which maps reduced degree of freedom representations of
        g into a full image
    """
    if out_shape is not None:
        assert mask.shape == out_shape, "mask shape does not match out_shape"
    labels, num = label(mask)
    sizes = np.sqrt([mask[labels == i].sum() for i in range(1, num+1)])
    rows, cols, vals = [], [], []
    count = 0  # start independent values after fixed values
    # g_i = \sum_j F_{i,j} \tilde{g}_j
    #TODO: MAKE SURE THESE INDICES ARE ALL CORRECT!
    for i, j in enumerate(labels.ravel()):
        rows.append(i)
        if j:
            cols.append(j - 1)
            vals.append(1 / sizes[j - 1])  # to prevent crazy scales
        else:
            cols.append(num + count)
            vals.append(1)
            count += 1
    return csr_matrix((vals, (rows, cols)), (mask.size, num + count))


class MyLinearOperator(LinearOperator):
    def __init__(self, shape, matvec, rmatvec=None):
        """
        This linear operator assumes 

        """
        if (shape[0] != shape[1]) and rmatvec is None:
            raise TypeError("Non-square matrix requires rmatvec_fn.")
        super(MyLinearOperator, self).__init__("float64", shape)
        self.matvec = matvec
        self.rmatvec = rmatvec if rmatvec is not None else matvec

    def _matvec(self, x):
        return self.matvec(x)

    def _rmatvec(self, x):
        return self.rmatvec(x)

    @property
    def T(self):
        return self.H
