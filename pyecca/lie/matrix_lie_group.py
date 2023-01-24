import casadi as ca
import abc


class MatrixLieGroup(abc.ABC):
    def __init__(self, group_params, algebra_params, group_shape):
        self.group_params = group_params
        self.algebra_params = algebra_params
        self.group_shape = group_shape

    def check_group_shape(self, a):
        assert a.shape == self.group_shape or a.shape == (self.group_shape[0],)

    @abc.abstractmethod
    def identity(self) -> ca.SX:
        ...

    @abc.abstractmethod
    def product(self, a, b):
        ...

    @abc.abstractmethod
    def inv(self, a) -> ca.SX:
        ...

    @abc.abstractmethod
    def exp(self, v) -> ca.SX:
        ...

    @abc.abstractmethod
    def log(self, a) -> ca.SX:
        ...
