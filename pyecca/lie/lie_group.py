import casadi as ca
import abc
from typing import Tuple


class LieGroup(abc.ABC):
    """
    This is a generic Lie Group class. It does NOT assume matrix
    lie groups. This is because matrix Lie groups often are not the
    most efficient way to implement group operations. For instance,
    SO3 could be represented by the Matrix Lie group of direction 
    cosine matrices, but with 4 elements, the product of quaternion,
    kinematics, exponential, log etc, can be more efficiently 
    implemented directly with the 4 quaternions instead of using all
    9 elements of the DCM.
    """
    
    def __init__(self, group_params: int, algebra_params: int, group_shape: Tuple(int)):
        """
        @param group_params: The number of parameters in the group, (e.g. for quaternion this would be 4,
        for DCM this would be 9)
        @param algebra_params: The number of parameters for the Lie algebra (e.g. for SO3 this would be 3, for
        SO2 this would be 1)
        @param group_shape: The shape of the Matrix Lie group and matrix Lie algebra
        """
        self.group_params = group_params
        self.algebra_params = algebra_params
        self.group_shape = group_shape

    def assert_size_group_shape(self, a):
        """
        Checks the group shape
        """
        assert a.shape == self.group_shape or a.shape == (self.group_shape[0],)

    def assert_size_group_params(self, a):
        """
        Checks the size of the group parameters
        """
        assert a.shape == (self.group_params,)

    def assert_size_algebra_params(self, a):
        """
        Checks the size of the algebra parameters
        """
        assert a.shape == (self.group_params,)

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

    @abc.abstractmethod
    def vee(self, X) -> ca.SX:
        ...

    @abc.abstractmethod
    def wedge(self, v) -> ca.SX:
        ...

    @abc.abstractmethod
    def ad(self, v) -> ca.SX:
        ...

    @abc.abstractmethod
    def Ad(self, v) -> ca.SX:
        ...