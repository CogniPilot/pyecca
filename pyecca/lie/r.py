import casadi as ca

from .lie_group import LieGroup


class R(LieGroup):
    """
    n dimensionoal translation group
    """
    
    def __init__(self, n):
        super().__init__(group_params=n, algebra_params=n, group_shape=(n, 1))

    def identity(self):
        return ca.DM([0]*self.group_params)

    def product(self, a, b):
        assert a.shape[0] == self.group_params
        assert b.shape[0] == self.group_params
        return a + b

    def inv(self, a):
        assert a.shape[0] == self.group_params
        return -a

    def exp(self, v):
        assert v.shape[0] == self.algebra_params
        return v

    def log(self, a):
        assert a.shape[0] == self.group_params
        return a
    
    def vee(self, X):
        v = ca.SX(self.group_params, 1)
        for i in range(self.group_params):
            v[i, 0] = X[i, self.group_params]
        return v

    def wedge(self, v):
        X = ca.SX.zeros(self.group_params, self.group_params)
        for i in range(self.group_params):
            X[i, self.group_params] = v[i, 0]
        return X

    def ad(self, X):
        

R2 = R(2)
R3 = R(3)