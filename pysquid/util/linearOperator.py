import numpy as np
from scipy.sparse.linalg import LinearOperator


class MyLinearOperator(LinearOperator):
    def __init__(self, shape, matvec, rmatvec=None):
        """
        This linear operator assumes 

        """
        if (shape[0] != shape[1]) and rmatvec is None:
            raise TypeError("Non-square matrix requires rmatvec_fn.")
        super(MyLinearOperator, self).__init__('float64', shape)
        self.matvec = matvec
        self.rmatvec = rmatvec if rmatvec is not None else matvec
    
    def _matvec(self, x):
        return self.matvec(x) 

    def _rmatvec(self, x):
        return self.rmatvec(x)

    @property
    def T(self):
        return self.H

