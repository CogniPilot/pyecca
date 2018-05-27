"""
A module for Modified Rodrigues Parameters (MRPs)
"""
import casadi as ca
from .quat import Quat
from .dcm import Dcm
from .so3 import wedge as skew  # SO3 wedge is the R^3 -> R^3 X R^3 skew operator

Expr = ca.SX


# noinspection PyPep8Naming
class Mrp(Expr):

    def __init__(self, *args):
        super().__init__(*args)
        assert self.shape == (3, 1)

    def __add__(self, other: 'Mrp') -> 'Mrp':
        assert isinstance(other, Mrp)
        return Mrp(Expr(self) + Expr(other))

    def __sub__(self, other: 'Mrp') -> 'Mrp':
        assert isinstance(other, Mrp)
        return Mrp(Expr(self) - Expr(other))

    def __neg__(self):
        return Mrp(-Expr(self))

    def __rmul__(self, other: Expr) -> 'Mrp':
        return Mrp(other * Expr(self))

    def __mul__(self, other: 'Mrp') -> 'Mrp':
        """
        Take the product of two MRPs, representing successive rotations such that:
        to_dcm(a)*to_dcm(b) = to_dcm(a*b)
        :param other: The other MRP, on the right.
        :return: The resultant MRP.
        """
        # order of a, b reversed from Schaub to match
        # definition of hamilton quaternion product so
        # that Dcm(a*b) = Dcm(a)*Dcm(b)
        assert isinstance(other, Mrp)
        a = self
        b = other
        na_sq = ca.dot(a, a)
        nb_sq = ca.dot(b, b)
        r = ((1 - na_sq) * b + (1 - nb_sq) * a - 2 * Mrp(ca.cross(b, a))) / (1 + na_sq * nb_sq - 2 * ca.dot(b, a))
        return Mrp(r)

    def inv(self) -> 'Mrp':
        """
        The multiplicative inverse for MRPs. It happens to be equal to the negative of the MRPs.
        :return: The inverse of the MRP.
        """
        return Mrp(-self)

    def shadow(self) -> 'Mrp':
        """
        Convert MRPs to their shadow (the MRPs corresponding to the quaternion with opposite sign).
        Both the MRPs and shadow MRPs represent the same attitude, but one of the two's magnitude is always less than 1,
        while the other's magnitude can approach inf near a rotation of 2*pi. So this function is used to switch to
        the other set when an MRP magnitude is greater than one to avoid the singularity.
        :return: The shadow MRP
        """
        a = self
        n_sq = ca.dot(a, a)
        return Mrp(-a / n_sq)

    def B(self) -> Expr:
        """
        A matrix used to compute the MRPs kinematics.
        :return: The B matrix.
        """
        a = self
        n_sq = ca.dot(a, a)
        X = skew(a)
        return 0.25 * ((1 - n_sq) * Expr.eye(3) + 2 * X + 2 * ca.mtimes(a, ca.transpose(a)))

    def derivative(self, w: Expr) -> Expr:
        """
        The kinematic equation relating the time derivative of MRPs given the current MRPs and the angular velocity.
        :param w: The angular velocity.
        :return: The time derivative of the MRPs.
        """
        return ca.mtimes(self.B(), w)

    def to_dcm(self) -> Dcm:
        """
        Convert to a DCM.
        :return: The DCM.
        """
        X = skew(self)
        n_sq = ca.dot(self, self)
        X_sq = ca.mtimes(X, X)
        R = Expr.eye(3) + (8 * X_sq - 4 * (1 - n_sq) * X) / (1 + n_sq) ** 2
        # return transpose, due to convention difference in book
        return Dcm(R.T)

    @classmethod
    def exp(cls, w: Expr) -> 'Mrp':
        """
        The exponential map from the Lie algebra element components to the Lie group.
        :param w: The Lie algebra, represented by components of an angular velocity vector.
        :return: The Lie group, represented by MRPs.
        """
        n = ca.norm_2(w)
        axis = ca.if_else(n < 1e-6, Expr([1, 0, 0]), w / n)
        angle = ca.mod(n, 2 * ca.pi)
        phi = ca.if_else(ca.fabs(angle) < ca.pi, angle, angle - ca.sign(angle) * 2 * ca.pi)
        return cls(ca.tan(phi / 4) * axis)

    @classmethod
    def log(cls, a: Expr) -> 'Mrp':
        """
        The inverse exponential map form the Lie group to the Lie algebra element components.
        :param a: The Lie group, represented by MRPs.
        :return: The Lie algebra, represented by an angular velocity vector.
        """
        n = ca.mod(ca.norm_2(a), 2 * ca.pi)
        axis = ca.if_else(n < 1e-6, Expr([1, 0, 0]), a / n)
        angle = 4 * ca.atan(n)
        phi = ca.if_else(ca.fabs(angle) < ca.pi, angle, angle - ca.sign(angle) * 2 * ca.pi)
        return cls(phi * axis)

    @classmethod
    def from_quat(cls, q: Quat) -> 'Mrp':
        """
        Convert from a quaternion to MRPs.
        :param q: The quaternion.
        :return: The MRPs.
        """
        assert isinstance(q, Quat)
        x = Expr(3, 1)
        x[0] = q[1] / (1 + q[0])
        x[1] = q[2] / (1 + q[0])
        x[2] = q[3] / (1 + q[0])
        return cls(x)

    @classmethod
    def from_dcm(cls, R: Dcm) -> 'Mrp':
        """
        Convert from a DCM to a MRPs.
        :param R: The DCM.
        :return: The MRPs.
        """
        return cls.from_quat(Quat.from_dcm(R))
