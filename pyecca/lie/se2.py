import casadi as ca
from typing import Tuple

from .matrix_lie_group import MatrixLieGroup
from .util import series_dict


class _SE2(MatrixLieGroup):
    def __init__(self):
        super().__init__(group_params=6, algebra_params=3, group_shape=(3, 3))

    def check_group_shape(self, a) -> None:
        assert a.shape == self.group_shape or a.shape == (self.group_shape[0],)

    def ad_matrix(self, v) -> ca.SX:
        """
        takes 3x1 lie algebra
        input vee operator [x,y,theta]
        """
        x = v[0]
        y = v[1]
        theta = v[2]
        ad_se2 = ca.SX(3, 3)
        ad_se2[0, 1] = -theta
        ad_se2[0, 2] = y
        ad_se2[1, 0] = theta
        ad_se2[1, 2] = -x
        return ad_se2

    def vee(self, X):
        """
        This takes in an element of the SE2 Lie Group (Wedge Form) and returns the se2 Lie Algebra elements
        """
        print(X.shape)
        v = ca.SX(3, 1)
        v[0] = X[0, 2]  # x
        v[1] = X[1, 2]  # y
        v[2] = X[1, 0]  # theta
        return v

    def wedge(self, v):
        """
        This takes in an element of the se2 Lie Algebra and returns the se2 Lie Algebra matrix
        """
        X = ca.SX.zeros(3, 3)
        x = v[0]
        y = v[1]
        theta = v[2]
        X[0, 1] = -theta
        X[1, 0] = theta
        X[0, 2] = x
        X[1, 2] = y
        return X

    def product(self, a, b):
        self.check_group_shape(a)
        self.check_group_shape(b)
        return a @ b

    def identity(self) -> ca.SX:
        return ca.SX.zeros(3)

    def exp(self, v):  # accept input in wedge operator form
        v = self.vee(v)
        theta = v[2]

        # translational components u
        u = ca.SX(2, 1)
        u[0] = v[0]
        u[1] = v[1]

        A = series_dict["sin(x)/x"](theta)
        B = series_dict["(1 - cos(x))/x"](theta)

        V = ca.SX(2, 2)
        V[0, 0] = A
        V[0, 1] = -B
        V[1, 0] = B
        V[1, 1] = A

        R = ca.SX(2, 2)  # Exp(wedge(theta))
        cos_theta = ca.cos(theta)
        sin_theta = ca.sin(theta)
        R[0, 0] = cos_theta
        R[0, 1] = -sin_theta
        R[1, 0] = sin_theta
        R[1, 1] = cos_theta

        horz = ca.horzcat(R, V @ u)
        lastRow = ca.SX([0, 0, 1]).T
        return ca.vertcat(horz, lastRow)

    def inv(self, a):  # input a matrix of ca.SX form
        self.check_group_shape(a)
        a_inv = ca.solve(
            a, ca.SX.eye(3)
        )  # Google Group post mentioned ca.inv() could take too long, and should explore solve function
        return ca.transpose(a)

    def log(self, G):
        theta = ca.atan(G[1, 0] / G[0, 0])

        # t is translational component vector
        t = ca.SX(2, 1)
        t[0] = G[0, 2]
        t[1] = G[1, 2]

        A = series_dict["sin(x)/x"](theta)
        B = series_dict["(1 - cos(x))/x"](theta)

        V_inv = ca.SX(2, 2)
        V_inv[0, 0] = A
        V_inv[0, 1] = B
        V_inv[1, 0] = -B
        V_inv[1, 1] = A
        V_inv = V_inv / (A**2 + B**2)

        vt_i = V_inv @ t

        return self.wedge(ca.vertcat(vt_i, theta))

    def diff_correction(self, v):  # U Matrix for se2 with input vee operator
        return ca.inv(self.diff_correction_inv(v))

    def diff_correction_inv(self, v):  # U_inv of se2 input vee operator

        # v = se2.vee(v)  #This only applies if v is inputed from Lie Group format

        theta = v[2]
        # X_so3 = se2.wedge(v) #wedge operator for so2 (required [x,y,theta])

        A = series_dict["sin(x)/x"]
        B = series_dict["(1 - cos(x))/x"]

        ad = self.ad_matrix(v)
        I = ca.SX_eye(3)
        u_inv = I + A * ad + B * (ad @ ad)
        return u_inv


SE2 = _SE2()
