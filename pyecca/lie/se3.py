import casadi as ca
from pyecca.lie import so3, se3

from .util import series_dict
from .matrix_lie_group import MatrixLieGroup
from .so3 import Quat, Euler, Mrp, Dcm


class _SE3(MatrixLieGroup):
    def __init__(self, SO3=None):
        if SO3 == None:
            self.SO3 = so3.Dcm
        else:
            self.SO3 = SO3
        super().__init__(
            group_params=3 + self.SO3.group_params,
            algebra_params=3 + self.SO3.algebra_params,
            group_shape=(4, 4),
        )

    def ad_matrix(self, v):
        """
        takes 6x1 lie algebra
        input vee operator [x,y,z,theta1,theta2,theta3]
        """
        ad_se3 = ca.SX(6, 6)
        ad_se3[0, 1] = -v[5]
        ad_se3[0, 2] = v[3]
        ad_se3[0, 4] = -v[2]
        ad_se3[0, 5] = v[1]
        ad_se3[1, 0] = v[5]
        ad_se3[1, 2] = -v[3]
        ad_se3[1, 3] = v[2]
        ad_se3[1, 5] = -v[0]
        ad_se3[2, 0] = -v[4]
        ad_se3[2, 1] = v[3]
        ad_se3[2, 3] = -v[1]
        ad_se3[2, 4] = v[0]
        ad_se3[3, 4] = -v[5]
        ad_se3[3, 5] = v[4]
        ad_se3[4, 3] = v[5]
        ad_se3[4, 5] = -v[3]
        ad_se3[5, 3] = -v[4]
        ad_se3[5, 4] = v[3]
        return ad_se3

    def vee(self, X):
        """
        This takes in an element of the SE3 Lie Group (Wedge Form) and returns the se3 Lie Algebra elements
        """
        v = ca.SX(6, 1)
        v[0, 0] = X[0, 3]  # x
        v[1, 0] = X[1, 3]  # y
        v[2, 0] = X[2, 3]  # z
        v[3, 0] = X[2, 1]  # theta0
        v[4, 0] = X[0, 2]  # theta1
        v[5, 0] = X[1, 0]  # theta2
        return v

    def wedge(self, v):
        """
        This takes in an element of the se3 Lie Algebra and returns the se3 Lie Algebra matrix

        v: [x,y,z,theta0,theta1,theta2]
        """
        X = ca.SX.zeros(4, 4)
        X[0, 3] = v[0]
        X[1, 3] = v[1]
        X[2, 3] = v[2]
        X[:3, :3] = so3.Dcm.wedge(v[3:6])
        return X

    def exp(self, v):  # accept input in wedge operator form
        v = self.vee(v)
        # v = [x,y,z,theta1,theta2,theta3]
        v_so3 = v[
            3:6
        ]  # grab only rotation terms for so3 uses ##corrected to v_so3 = v[3:6]
        X_so3 = so3.Dcm.wedge(v_so3)  # wedge operator for so3
        theta = ca.norm_2(
            so3.Dcm.vee(X_so3)
        )  # theta term using norm for sqrt(theta1**2+theta2**2+theta3**2)

        # translational components u
        u = ca.SX(3, 1)
        u[0, 0] = v[0]
        u[1, 0] = v[1]
        u[2, 0] = v[2]

        R = so3.Dcm.exp(
            v_so3
        )  #'Dcm' for direction cosine matrix representation of so3 LieGroup Rotational

        A = series_dict["sin(x)/x"](theta)
        B = series_dict["(1 - cos(x))/x^2"](theta)
        C = (1 - A) / theta**2

        V = ca.SX.eye(3) + B * X_so3 + C * X_so3 @ X_so3

        horz = ca.horzcat(R, ca.mtimes(V, u))

        lastRow = ca.SX([0, 0, 0, 1]).T

        return ca.vertcat(horz, lastRow)

    def identity(self):
        return ca.SX.eye(4)

    def product(self, a, b):
        self.check_group_shape(a)
        self.check_group_shape(b)
        return a @ b

    def inv(self, a):  # input a matrix of SX form from casadi
        self.check_group_shape(a)
        a_inv = ca.solve(
            a, ca.SX.eye(6)
        )  # Google Group post mentioned ca.inv() could take too long, and should explore solve function
        return ca.transpose(a)

    def log(self, G):
        R = G[:3, :3]
        theta = ca.arccos((ca.trace(R) - 1) / 2)
        wSkew = so3.Dcm.wedge(so3.Dcm.log(R))
        A = series_dict["sin(x)/x"](theta)
        B = series_dict["(1 - cos(x))/x^2"](theta)
        V_inv = (
            ca.SX.eye(3)
            - wSkew / 2
            + (1 / theta**2) * (1 - A / (2 * B)) * wSkew @ wSkew
        )

        t = ca.SX(3, 1)
        t[0] = G[0, 3]
        t[1] = G[1, 3]
        t[2] = G[2, 3]

        uInv = V_inv @ t
        horz2 = ca.horzcat(wSkew, uInv)
        lastRow2 = ca.SX([0, 0, 0, 0]).T
        return ca.vertcat(horz2, lastRow2)

    def diff_correction(self, v):  # U Matrix for se3 with input vee operator
        return ca.inv(self.diff_correction_inv(v))

    def diff_correction_inv(self, v):  # U_inv of se3 input vee operator
        # v = se3.vee(v)  #This only applies if v is inputed from Lie Group format

        v_so3 = v[
            3:6
        ]  # grab only rotation terms for so3 uses ## changed to match NASAULI paper order of vee v[3:6]
        X_so3 = so3.wedge(v_so3)  # wedge operator for so3
        theta = ca.norm_2(
            so3.vee(X_so3)
        )  # theta term using norm for sqrt(theta1**2+theta2**2+theta3**2)

        A = series_dict["sin(x)/x"]
        B = series_dict["(1 - cos(x))/x^2"]
        C = series_dict["(x - sin(x))/x^3"]

        ad = se3.ad_matrix(v)
        I = ca.SX_eye(6)
        u_inv = I + c2 * ad + c3 * se3.matmul(ad, ad)
        return u_inv

        # u_inv = ca.SX(6, 6)
        # u1 = c2*(-v[4]**2 - v[5]**2) + 1
        # u2 = -c1*v[5] + c2*v[3]*v[4]
        # u3 = c1 * v[4] + c2 * v[3]*v[5]
        # u4 = c2 * (-2*v[4]*v[1]-2*v[5]*v[2])
        # u5 = -c1 * v[2] + c2*(v[4]*v[0]+v[3]*v[1])
        # u6 = c1 * v[1] + c2*(v[3]*v[2]+v[5]*v[0])
        # uInvR1 = ca.vertcat(u1,u2,u3,u4,u5,u6)

        # u1 = c1 * v[5] + c2 * v[3] * v[4]
        # u2 = c2 *(-v[3]**2 - v[5]**2)+1
        # u3 = -c1*v[3] + c2 * v[4]*v[5]
        # u4 = c1 * v[2] + c2 * (v[3]*v[1]+v[4]*v[0])
        # u5 = c2* (-2*v[3] * v[0] -2*v[5]*v[2])
        # u6 = -c1 * v[0] + c2 * (v[4]*v[2] + v[5] *v[1])
        # uInvR2 = ca.vertcat(u1,u2,u3,u4,u5,u6)

        # u1 = -c1 * v[4] + c2 * v[3] * v[5]
        # u2 = c1 * v[3] + c2 * v[4] * v[5]
        # u3 = c1 * (-v[3] **2  - v[4]**2) +1
        # u4 = -c1 * v[1] + c2 * (v[3]*v[2] + v[5]*v[0])
        # u5 = c1 * v[0] + c2 * (v[4]*v[2] + v[5] *v[1])
        # u6 = c2 * (-2*v[3]*v[0] - 2*v[4] *v[1])
        # uInvR3 = ca.vertcat(u1,u2,u3,u4,u5,u6)

        # u1 = 0
        # u2 = 0
        # u3 = 0
        # u4 = c2 * (- v[4]**2 - v[5]**2) +1
        # u5 = -c1*v[5] + c2*v[3]*v[4]
        # u6 = c1 * v[4] + c2 * v[3] * v[5]
        # uInvR4 = ca.vertcat(u1,u2,u3,u4,u5,u6)

        # u1 = 0
        # u2 = 0
        # u3 = 0
        # u4 = c1 * v[5] + c2 * v[3] * v[4]
        # u5 = c2 * (-v[3]**2 - v[5]**2) +1
        # u6 = -c1 * v[3] + c2 * v[4] *v[5]
        # uInvR5 = ca.vertcat(u1,u2,u3,u4,u5,u6)

        # u1 = 0
        # u2 = 0
        # u3 = 0
        # u4 = -c1 * v[4] + c2 * v[3] * v[5]
        # u5 = c1 * v[3] + c2 * v[4] * v[5]
        # u6 = c2 * (-v[3] **2 - v[4]**2)+1
        # uInvR6 = ca.vertcat(u1,u2,u3,u4,u5,u6)

        # u_inv = ca.transpose(ca.horzcat(uInvR1,uInvR2,uInvR3,uInvR4, uInvR5, uInvR6))
        # return u_inv

    # verify this with series solution

    # https://github.com/jgoppert/pyecca/blob/master/pyecce/estimation/attitude/algorithms/mrp.py
    # Use this to try to get casadi to draw a plot for this
    # line 112-116 help for drawing plots

    # https://github.com/jgoppert/pyecca/blob/master/pyecca/estimation/attitude/algorithms/common.py
    # This import to import needed casadi command

    # New notes (Oct 18 22)
    ## sympy.cse(f) to find common self expression using sympy to clean up the casadi plot
    # cse_def, cse_expr = sympy.cse(f)

    # Oct 25 Update
    # work on updating SE2 umatrix to casadi
    # get SE2 U matrix
    # u matrix can be found through casadi using inverse function for casadi


SE3Dcm = _SE3(Dcm)
SE3Euler = _SE3(Euler)
SE3Quat = _SE3(Quat)
SE3Mrp = _SE3(Mrp)
