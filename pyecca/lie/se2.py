import casadi as ca
from typing import Tuple

from .matrix_lie_group import MatrixLieGroup
from .util import series_dict


class _SE2(MatrixLieGroup):
    def __init__(self):
        super().__init__(group_params=3, algebra_params=3, group_shape=(3, 3))

    def check_group_shape(self, a) -> None:
        assert a.shape == self.group_shape or a.shape == (self.group_shape[0],)

    def check_param_shape(self, p) -> None:
        assert p.shape == (self.group_params, 1) or p.shape == (self.group_params,)

    def R(self, p) -> ca.SX:
        """
        Returns embedded SO(2) rotation group
        """
        self.check_param_shape(p)
        theta = p[2]
        cth = ca.cos(theta)
        sth = ca.sin(theta)
        R = ca.SX.zeros(2, 2)
        R[0, 0] = cth
        R[0, 1] = -sth
        R[1, 0] = sth
        R[1, 1] = cth
        return R
    
    def v(self, p) -> ca.SX:
        """
        Returns embedded R2 vector
        """
        self.check_param_shape(p)
        v = ca.SX.zeros(2)
        v[0] = p[0]
        v[1] = p[1]
        return v
        
    def ad_matrix(self, v) -> ca.SX:
        """
        takes 3x1 lie algebra
        input vee operator [x,y,theta]
        """
        x = v[0]
        y = v[1]
        theta = v[2]
        ad_se2 = ca.SX.zeros(3, 3)
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
        v = ca.SX.zeros(3)
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

    def product(self, p1, p2):
        self.check_param_shape(p1)
        self.check_param_shape(p2)
        x1 = p1[0]
        y1 = p1[1]
        theta1 = p1[2]
        x2 = p2[0]
        y2 = p2[1]
        theta2 = p2[2]
        p3 = ca.SX.zeros(3)
        theta3 = theta1 + theta2
        
        theta3 = ca.atan2(ca.sin(theta3), ca.cos(theta3))
                
        v3 = self.R(p1)@self.v(p2) + self.v(p1)
        p3[0] = v3[0]
        p3[1] = v3[1]
        p3[2] = theta3;
        return p3

    def identity(self) -> ca.SX:
        return ca.SX.zeros(3)

    def exp(self, v):  # accept input in wedge operator form
        v = self.vee(v)
        theta = v[2]

        # translational components u
        u = ca.SX.zeros(2)
        u[0] = v[0]
        u[1] = v[1]

        A = series_dict["sin(x)/x"](theta)
        B = series_dict["(1 - cos(x))/x"](theta)

        V = ca.SX.zeros(2, 2)
        V[0, 0] = A
        V[0, 1] = -B
        V[1, 0] = B
        V[1, 1] = A

        R = ca.SX.zeros(2, 2)  # Exp(wedge(theta))
        cos_theta = ca.cos(theta)
        sin_theta = ca.sin(theta)
        R[0, 0] = cos_theta
        R[0, 1] = -sin_theta
        R[1, 0] = sin_theta
        R[1, 1] = cos_theta

        horz = ca.horzcat(R, V @ u)
        lastRow = ca.SX([0, 0, 1]).T
        return ca.vertcat(horz, lastRow)

    def inv(self, p):  # group parameters (x, y, theta)
        self.check_param_shape(p)
        theta = p[2]
        v1 = ca.SX.zeros(2)
        v1[0] = p[0]
        v1[1] = p[1]
        v2 = -self.R(p).T@v1
        p_inv = ca.SX.zeros(3)
        p_inv[0] = v2[0]
        p_inv[1] = v2[1]
        p_inv[2] = -theta
        return p_inv

    def log(self, p):
        v = self.v(p)

        theta = p[2]
        A = series_dict["sin(x)/x"](theta)
        B = series_dict["(1 - cos(x))/x"](theta)

        V_inv = ca.SX.zeros(2, 2)
        V_inv[0, 0] = A
        V_inv[0, 1] = B
        V_inv[1, 0] = -B
        V_inv[1, 1] = A
        V_inv = V_inv / (A**2 + B**2)

        vt_i = V_inv @ v
        u = ca.SX.zeros(3)
        u[0] = vt_i[0]
        u[1] = vt_i[1]
        u[2] = p[2]
        return u

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
