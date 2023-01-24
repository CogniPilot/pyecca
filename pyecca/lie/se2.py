import casadi as ca
from typing import Tuple

from .matrix_lie_group import MatrixLieGroup
from .util import eps


class _SE2(MatrixLieGroup):

    def __init__(self):
        super().__init__(
            groups_params=6,
            algebra_params=3,
            group_shape=(3, 3))

    def check_group_shape(self, a) -> None:
        assert a.shape == self.group_shape or a.shape == (self.group_shape[0],)

    def ad_matrix(self, v) -> ca.SX:
        """
        takes 3x1 lie algebra
        input vee operator [x,y,theta]
        """
        ad_se2 = ca.SX(3, 3)
        ad_se2[0, 1] = -v[2]
        ad_se2[0, 2] = v[1]
        ad_se2[1, 0] = v[2]
        ad_se2[1, 2] = -v[0]
        return ad_se2

    def vee(self, X):
        """
        This takes in an element of the SE2 Lie Group (Wedge Form) and returns the se2 Lie Algebra elements
        """
        v = ca.SX(3, 1)  # [x,y,theta]
        v[0, 0] = -X[1, 2]
        v[1, 0] = X[0, 2]
        v[2, 0] = X[1, 0]
        return v

    def wedge(self, v):  # input v = [x,y,theta]
        """
        This takes in an element of the se2 Lie Algebra and returns the se2 Lie Algebra matrix
        """
        X = ca.SX.zeros(3, 3)
        X[0, 1] = -v[2]
        X[1, 0] = v[2]
        X[0, 2] = v[0]
        X[1, 2] = v[1]
        return X

    def matmul(self, a, b):
        self.check_group_shape(a)
        self.check_group_shape(b)
        return ca.mtimes(a, b)

    def exp(self, v):  # accept input in wedge operator form
        v = self.vee(v)
        theta = v[2]

        # translational components u
        u = ca.SX(2, 1)
        u[0, 0] = v[0]
        u[1, 0] = v[1]

        if type(v[1]) == "int" and theta < eps:
            a = 1 - theta**2 / 6 + theta**4 / 120
            b = 0.5 - theta**2 / 24 + theta**4 / 720
        else:
            a = ca.sin(theta) / theta
            b = 1 - ca.cos(theta) / theta

        V = ca.SX(2, 2)
        V[0, 0] = a
        V[0, 1] = -b
        V[1, 0] = b
        V[1, 1] = a

        if type(v[1]) == "int" and theta < eps:
            a = theta - theta**3 / 6 + theta**5 / 120
            b = 1 - theta**2 / 2 + theta**4 / 24
        else:
            a = ca.sin(theta)
            b = ca.cos(theta)

        R = ca.SX(2, 2)  # Exp(wedge(theta))
        R[0, 0] = b
        R[0, 1] = -a
        R[1, 0] = a
        R[1, 1] = b

        horz = ca.horzcat(R, ca.mtimes(V, u))

        lastRow = ca.horzcat(0, 0, 1)

        return ca.vertcat(horz, lastRow)

    def one(self):
        return ca.SX.zeros(3, 1)

    def inv(self, a):  # input a matrix of ca.SX form
        self.check_group_shape(a)
        a_inv = ca.solve(
            a, ca.SX.eye(3)
        )  # Google Group post mentioned ca.inv() could take too long, and should explore solve function
        return ca.transpose(a)

    def log(self, G):
        theta = ca.arccos(
            ((G[0, 0] + G[1, 1]) - 1) / 2
        )  # RECHECK (Unsure of where this comes from)
        wSkew = self.wedge(G[:2, :2])

        # t is translational component vector
        t = ca.SX(3, 1)
        t[0, 0] = G[0, 2]
        t[1, 0] = G[1, 2]

        if type(v[1]) == "int" and theta < eps:
            a = 1 - theta**2 / 6 + theta**4 / 120
            b = 0.5 - theta**2 / 24 + theta**4 / 720
        else:
            a = ca.sin(theta) / theta
            b = 1 - ca.cos(theta) / theta

        V_inv = ca.SX(2, 2)
        V_inv[0, 0] = a
        V_inv[0, 1] = b
        V_inv[1, 0] = -b
        V_inv[1, 1] = a
        V_inv = V_inv / (a**2 + b**2)

        vt_i = ca.mtimes(V_inv, t)
        t_term = theta  # Last Row for se2

        return ca.vertcat(vt_i, t_term)

    def diff_correction(self, v):  # U Matrix for se2 with input vee operator
        return ca.inv(self.diff_correction_inv(v))

    def diff_correction_inv(self, v):  # U_inv of se2 input vee operator

        # v = se2.vee(v)  #This only applies if v is inputed from Lie Group format

        theta = v[2]
        # X_so3 = se2.wedge(v) #wedge operator for so2 (required [x,y,theta])

        if type(v[1]) == "int" and theta < eps:
            c1 = 1 - theta**2 / 6 + theta**4 / 120
            c2 = 0.5 - theta**2 / 24 + theta**4 / 720
        else:
            c1 = ca.sin(theta) / theta
            c2 = 1 - ca.cos(theta) / theta

        ad = self.ad_matrix(v)
        I = ca.SX_eye(3)
        u_inv = I + c1 * ad + c2 * (ad@ad)
        return u_inv

SE2 = _SE2()