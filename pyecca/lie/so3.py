import casadi as ca

from .matrix_lie_group import MatrixLieGroup
from .util import series_dict


# see: https://ethaneade.com/lie.pdf


EPS = 1e-7


class _SO3Base(MatrixLieGroup):
    def vee(self, X):
        v = ca.SX(3, 1)
        v[0, 0] = X[2, 1]
        v[1, 0] = X[0, 2]
        v[2, 0] = X[1, 0]
        return v

    def wedge(self, v):
        X = ca.SX(3, 3)
        theta0 = v[0]
        theta1 = v[1]
        theta2 = v[2]
        X[0, 1] = -theta2
        X[0, 2] = theta1
        X[1, 0] = theta2
        X[1, 2] = -theta0
        X[2, 0] = -theta1
        X[2, 1] = theta0
        return X


class _Dcm(_SO3Base):
    def __init__(self):
        super().__init__(group_params=9, algebra_params=3, group_shape=(3, 3))

    def identity(self) -> ca.SX:
        return ca.SX.eye(3)

    def product(self, a, b):
        self.check_group_shape(a)
        self.check_group_shape(b)
        return a @ b

    def inv(self, a):
        self.check_group_shape(a)
        return ca.transpose(a)

    def exp(self, v):
        theta = ca.norm_2(v)
        X = self.wedge(v)
        A = series_dict["sin(x)/x"]
        B = series_dict["(1 - cos(x))/x^2"]
        return ca.SX.eye(3) + A(theta) * X + B(theta) * X @ X

    def log(self, R):
        theta = ca.arccos((ca.trace(R) - 1) / 2)
        A = series_dict["sin(x)/x"]
        return self.vee((R - R.T) / (A(theta) * 2))

    def kinematics(self, R, w):
        assert R.shape == (3, 3)
        assert w.shape == (3, 1)
        return R @ self.wedge(w)

    def from_quat(self, q):
        assert q.shape == (4, 1)
        R = ca.SX(3, 3)
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

    def from_mrp(self, r):
        assert r.shape == (4, 1)
        a = r[:3]
        X = self.wedge(a)
        n_sq = ca.dot(a, a)
        X_sq = X @ X
        R = ca.SX.eye(3) + (8 * X_sq - 4 * (1 - n_sq) * X) / (1 + n_sq) ** 2
        # return transpose, due to convention difference in book
        return R.T

    def from_euler(self, e):
        return self.from_quat(Quat.from_euler(e))


Dcm = _Dcm()


class _Mrp(_SO3Base):
    def __init__(self):
        super().__init__(group_params=4, algebra_params=3, group_shape=(4, 1))

    def product(self, r1, r2):
        assert r1.shape == (4, 1) or r1.shape == (4,)
        assert r2.shape == (4, 1) or r2.shape == (4,)
        a = r1[:3]
        b = r2[:3]
        na_sq = ca.dot(a, a)
        nb_sq = ca.dot(b, b)
        res = ca.SX(4, 1)
        den = 1 + na_sq * nb_sq - 2 * ca.dot(b, a)
        res[:3] = ((1 - na_sq) * b + (1 - nb_sq) * a - 2 * ca.cross(b, a)) / den
        res[3] = 0  # shadow state
        return res

    def inv(self, r):
        assert r.shape == (4, 1) or r.shape == (4,)
        return ca.vertcat(-r[:3], r[3])

    def exp(self, v):
        assert v.shape == (3, 1) or v.shape == (3,)
        angle = ca.norm_2(v)
        res = ca.SX(4, 1)
        res[:3] = ca.tan(angle / 4) * v / angle
        res[3] = 0
        return ca.if_else(angle > EPS, res, ca.SX([0, 0, 0, 0]))

    def log(self, r):
        assert r.shape == (4, 1) or r.shape == (4,)
        n = ca.norm_2(r[:3])
        return ca.if_else(n > EPS, 4 * ca.atan(n) * r[:3] / n, ca.SX([0, 0, 0]))

    def shadow(self, r):
        assert r.shape == (4, 1) or r.shape == (4,)
        n_sq = ca.dot(r[:3], r[:3])
        res = ca.SX(4, 1)
        res[:3] = -r[:3] / n_sq
        res[3] = ca.logic_not(r[3])
        return res

    def shadow_if_necessary(self, r):
        assert r.shape == (4, 1) or r.shape == (4,)
        return ca.if_else(ca.norm_2(r[:3]) > 1, self.shadow(r), r)

    def kinematics(self, r, w):
        assert r.shape == (4, 1) or r.shape == (4,)
        assert w.shape == (3, 1) or w.shape == (3,)
        a = r[:3]
        n_sq = ca.dot(a, a)
        X = self.wedge(a)
        B = 0.25 * ((1 - n_sq) * ca.SX.eye(3) + 2 * X + 2 * a @ a.T)
        return ca.vertcat(B @ w, 0)

    def from_quat(self, q):
        assert q.shape == (4, 1) or q.shape == (4,)
        x = ca.SX(4, 1)
        den = 1 + q[0]
        x[0] = q[1] / den
        x[1] = q[2] / den
        x[2] = q[3] / den
        x[3] = 0
        r = self.shadow_if_necessary(x)
        r[3] = 0
        return r

    def from_dcm(self, R):
        return self.from_quat(Quat.from_dcm(R))

    def from_euler(self, e):
        return self.from_quat(Quat.from_euler(e))

    def identity(self) -> ca.SX:
        return ca.SX([0, 0, 0, 0])


Mrp = _Mrp()


class _Quat(_SO3Base):
    def __init__(self):
        super().__init__(group_params=4, algebra_params=3, group_shape=(4, 1))

    def identity(self) -> ca.SX:
        return ca.SX([1, 0, 0, 0])

    def product(self, a, b):
        assert a.shape == (4, 1) or a.shape == (4,)
        assert b.shape == (4, 1) or b.shape == (4,)
        r1 = a[0]
        v1 = a[1:]
        r2 = b[0]
        v2 = b[1:]
        res = ca.SX(4, 1)
        res[0] = r1 * r2 - ca.dot(v1, v2)
        res[1:] = r1 * v2 + r2 * v1 + ca.cross(v1, v2)
        return res

    def inv(self, q):
        assert q.shape == (4, 1) or q.shape == (4,)
        qi = ca.SX(4, 1)
        n = ca.norm_2(q)
        qi[0] = q[0] / n
        qi[1] = -q[1] / n
        qi[2] = -q[2] / n
        qi[3] = -q[3] / n
        return qi

    def exp(self, v):
        assert v.shape == (3, 1) or q.shape == (3,)
        q = ca.SX(4, 1)
        theta = ca.norm_2(v)
        q[0] = ca.cos(theta / 2)
        c = ca.sin(theta / 2)
        n = ca.norm_2(v)
        q[1] = c * v[0] / n
        q[2] = c * v[1] / n
        q[3] = c * v[2] / n
        return ca.if_else(n > EPS, q, ca.SX([1, 0, 0, 0]))

    def log(self, q):
        assert q.shape == (4, 1) or q.shape == (4,)
        v = ca.SX(3, 1)
        norm_q = ca.norm_2(q)
        theta = 2 * ca.acos(q[0])
        c = ca.sin(theta / 2)
        v[0] = theta * q[1] / c
        v[1] = theta * q[2] / c
        v[2] = theta * q[3] / c
        return ca.if_else(ca.fabs(c) > EPS, v, ca.SX([0, 0, 0]))

    def kinematics(self, q, w):
        """
        The kinematic equation relating the time derivative of quat given the current quat and the angular velocity
        in the body frame.
        :param q: The quaternion
        :param w: The angular velocity in the body frame.
        :return: The time derivative of the quat.
        """
        assert q.shape == (4, 1) or q.shape == (4,)
        assert w.shape == (3, 1) or w.shape == (3,)
        v = ca.SX(4, 1)
        v[0] = 0
        v[1] = w[0]
        v[2] = w[1]
        v[3] = w[2]
        return 0.5 * self.product(q, v)

    def from_mrp(self, r):
        assert r.shape == (4, 1) or r.shape == (4,)
        a = r[:3]
        q = ca.SX(4, 1)
        n_sq = ca.dot(a, a)
        den = 1 + n_sq
        q[0] = (1 - n_sq) / den
        for i in range(3):
            q[i + 1] = 2 * a[i] / den
        return ca.if_else(r[3], -q, q)

    def from_dcm(self, R):
        assert R.shape == (3, 3)
        b1 = 0.5 * ca.sqrt(1 + R[0, 0] + R[1, 1] + R[2, 2])
        b2 = 0.5 * ca.sqrt(1 + R[0, 0] - R[1, 1] - R[2, 2])
        b3 = 0.5 * ca.sqrt(1 - R[0, 0] + R[1, 1] - R[2, 2])
        b4 = 0.5 * ca.sqrt(1 - R[0, 0] - R[1, 1] + R[2, 2])

        q1 = ca.SX(4, 1)
        q1[0] = b1
        q1[1] = (R[2, 1] - R[1, 2]) / (4 * b1)
        q1[2] = (R[0, 2] - R[2, 0]) / (4 * b1)
        q1[3] = (R[1, 0] - R[0, 1]) / (4 * b1)

        q2 = ca.SX(4, 1)
        q2[0] = (R[2, 1] - R[1, 2]) / (4 * b2)
        q2[1] = b2
        q2[2] = (R[0, 1] + R[1, 0]) / (4 * b2)
        q2[3] = (R[0, 2] + R[2, 0]) / (4 * b2)

        q3 = ca.SX(4, 1)
        q3[0] = (R[0, 2] - R[2, 0]) / (4 * b3)
        q3[1] = (R[0, 1] + R[1, 0]) / (4 * b3)
        q3[2] = b3
        q3[3] = (R[1, 2] + R[2, 1]) / (4 * b3)

        q4 = ca.SX(4, 1)
        q4[0] = (R[1, 0] - R[0, 1]) / (4 * b4)
        q4[1] = (R[0, 2] + R[2, 0]) / (4 * b4)
        q4[2] = (R[1, 2] + R[2, 1]) / (4 * b4)
        q4[3] = b4

        q = ca.if_else(
            ca.trace(R) > 0,
            q1,
            ca.if_else(
                ca.logic_and(R[0, 0] > R[1, 1], R[0, 0] > R[2, 2]),
                q2,
                ca.if_else(R[1, 1] > R[2, 2], q3, q4),
            ),
        )
        return q

    def from_euler(self, e):
        assert e.shape == (3, 1) or e.shape == (3,)
        q = ca.SX(4, 1)
        cosPhi_2 = ca.cos(e[0] / 2)
        cosTheta_2 = ca.cos(e[1] / 2)
        cosPsi_2 = ca.cos(e[2] / 2)
        sinPhi_2 = ca.sin(e[0] / 2)
        sinTheta_2 = ca.sin(e[1] / 2)
        sinPsi_2 = ca.sin(e[2] / 2)
        q[0] = cosPhi_2 * cosTheta_2 * cosPsi_2 + sinPhi_2 * sinTheta_2 * sinPsi_2
        q[1] = sinPhi_2 * cosTheta_2 * cosPsi_2 - cosPhi_2 * sinTheta_2 * sinPsi_2
        q[2] = cosPhi_2 * sinTheta_2 * cosPsi_2 + sinPhi_2 * cosTheta_2 * sinPsi_2
        q[3] = cosPhi_2 * cosTheta_2 * sinPsi_2 - sinPhi_2 * sinTheta_2 * cosPsi_2
        return q


Quat = _Quat()


class _Euler(_SO3Base):
    def __init__(self):
        super().__init__(group_params=3, algebra_params=3, group_shape=(3, 1))

    def inv(self, e):
        return Euler.from_dcm(Dcm.inv(Dcm.from_euler(e)))

    def exp(self, v):
        return Euler.from_dcm(Dcm.exp(v))

    def log(self, e):
        return Dcm.log(Dcm.from_euler(e))

    def product(self, a, b):
        return Euler.from_dcm(Dcm.from_euler(a) @ Dcm.from_euler(b))

    def identity(self) -> ca.SX:
        return ca.SX([0, 0, 0])

    def from_quat(self, q):
        assert q.shape == (4, 1) or q.shape == (4,)
        e = ca.SX(3, 1)
        a = q[0]
        b = q[1]
        c = q[2]
        d = q[3]
        e[0] = ca.atan2(2 * (a * b + c * d), 1 - 2 * (b**2 + c**2))
        e[1] = ca.asin(2 * (a * c - d * b))
        e[2] = ca.atan2(2 * (a * d + b * c), 1 - 2 * (c**2 + d**2))
        return e

    def from_dcm(self, R):
        assert R.shape == (3, 3)
        return self.from_quat(Quat.from_dcm(R))

    def from_mrp(self, a):
        assert a.shape == (4, 1) or a.shape == (4,)
        return self.from_quat(Quat.from_mrp(a))


Euler = _Euler()
