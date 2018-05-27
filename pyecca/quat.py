"""
A module for quaternions (Euler parameters)
"""

import casadi as ca
from .dcm import Dcm

Expr = ca.SX


# noinspection PyPep8Naming
class Quat(Expr):

    def __init__(self, *args):
        if len(args) == 0:
            super().__init__(4, 1)
        else:
            super().__init__(*args)
        assert self.shape == (4, 1)

    def __add__(self, other: 'Quat') -> 'Quat':
        assert isinstance(other, Quat)
        return Quat(Expr(self) + Expr(other))

    def __sub__(self, other: 'Quat') -> 'Quat':
        assert isinstance(other, Quat)
        return Quat(Expr(self) - Expr(other))

    def __neg__(self):
        return Quat(-Expr(self))

    def __rmul__(self, other: Expr) -> 'Quat':
        return Quat(Expr(other) * Expr(self))

    def __mul__(self, other: 'Quat') -> 'Quat':
        """
        The product of two quaternions using the hamilton
        convention, so that Dcm(A)*Dcm(B) = Dcm(A*B).
        :param other: The second quaternion.
        :return: The quaternion product.
        """
        assert isinstance(other, Quat)
        a = self
        b = other
        r1 = a[0]
        v1 = a[1:]
        r2 = b[0]
        v2 = b[1:]
        res = Quat()
        res[0] = r1 * r2 - ca.dot(v1, v2)
        res[1:] = r1 * v2 + r2 * v1 + ca.cross(v1, v2)
        return res

    @classmethod
    def from_dcm(cls, R: Dcm) -> Expr:
        """
        Converts a direction cosine matrix to a quaternion.
        :param R: A direction cosine matrix.
        :return: The quaternion.
        """
        b1 = 0.5 * ca.sqrt(1 + R[0, 0] + R[1, 1] + R[2, 2])
        b2 = 0.5 * ca.sqrt(1 + R[0, 0] - R[1, 1] - R[2, 2])
        b3 = 0.5 * ca.sqrt(1 - R[0, 0] + R[1, 1] - R[2, 2])
        b4 = 0.5 * ca.sqrt(1 - R[0, 0] - R[1, 1] - R[2, 2])

        q1 = cls()
        q1[0] = b1
        q1[1] = (R[2, 1] - R[1, 2]) / (4 * b1)
        q1[2] = (R[0, 2] - R[2, 0]) / (4 * b1)
        q1[3] = (R[1, 0] - R[0, 1]) / (4 * b1)

        q2 = cls()
        q2[0] = (R[2, 1] - R[1, 2]) / (4 * b2)
        q2[1] = b2
        q2[2] = (R[0, 1] + R[1, 0]) / (4 * b2)
        q2[3] = (R[0, 2] + R[2, 0]) / (4 * b2)

        q3 = cls()
        q3[0] = (R[0, 2] - R[2, 0]) / (4 * b3)
        q3[1] = (R[0, 1] + R[1, 0]) / (4 * b3)
        q3[2] = b3
        q3[3] = (R[1, 2] + R[2, 1]) / (4 * b3)

        q4 = cls()
        q4[0] = (R[1, 0] - R[0, 1]) / (4 * b4)
        q4[1] = (R[0, 2] + R[2, 0]) / (4 * b4)
        q4[2] = (R[1, 2] + R[2, 1]) / (4 * b4)
        q4[3] = b4

        q = ca.if_else(
            R[0, 0] > 0,
            ca.if_else(R[1, 1] > 0, q1, q2),
            ca.if_else(R[1, 1] > R[2, 2], q3, q4)
        )
        return Quat(q)

    def to_dcm(self):
        """
        Converts a quaternion to a DCM.
        :return: The DCM.
        """
        q = self
        R = Expr(3, 3)
        a = q[0]
        b = q[1]
        c = q[2]
        d = q[3]
        aa = a * a
        ab = a * b
        ac = a * c
        ad = a * d
        bb = b * b
        bc = b * c
        bd = b * d
        cc = c * c
        cd = c * d
        dd = d * d
        R[0, 0] = aa + bb - cc - dd
        R[0, 1] = 2 * (bc - ad)
        R[0, 2] = 2 * (bd + ac)
        R[1, 0] = 2 * (bc + ad)
        R[1, 1] = aa + cc - bb - dd
        R[1, 2] = 2 * (cd - ab)
        R[2, 0] = 2 * (bd - ac)
        R[2, 1] = 2 * (cd + ab)
        R[2, 2] = aa + dd - bb - cc
        return R

    def to_euler(self) -> Expr:
        """
        Converts a quaternion to B321 Euler angles.
        :return: The B321 Euler angles (Phi [roll], Theta [pitch], Psi[heading])
        """
        q = self
        e = Expr(3, 1)
        a = q[0]
        b = q[1]
        c = q[2]
        d = q[3]
        e[0] = ca.atan2(2 * (a * b + c * d), 1 - 2 * (b ** 2 + c ** 2))
        e[1] = ca.asin(2 * (a * c - d * b))
        e[2] = ca.atan2(2 * (a * d + b * c), 1 - 2 * (c ** 2 + d ** 2))
        return e
