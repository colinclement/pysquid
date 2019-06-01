import numpy as np
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
    dh = sps.coo_matrix((hdata, (hrow, hcol)), shape=(N, N)).tocsr()
    dv = sps.coo_matrix((vdata, (vrow, vcol)), shape=(N, N)).tocsr()
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
    d2h = sps.coo_matrix((hdata, (hrow, hcol)), shape=(N, N)).tocsr()
    d2v = sps.coo_matrix((vdata, (vrow, vcol)), shape=(N, N)).tocsr()
    return d2h, d2v


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
