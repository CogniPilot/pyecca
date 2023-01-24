import casadi as ca

from .matrix_lie_group import MatrixLieGroup


class _R3(MatrixLieGroup):
    def __init__(self):
        super().__init__(group_params=3, algebra_params=3, group_shape=(3, 1))

    def identity(self):
        return ca.DM([0, 0, 0])

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


R3 = _R3()
