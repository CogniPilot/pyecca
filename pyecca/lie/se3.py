from . import so3

import casadi as ca

eps = 1e-7 # to avoid divide by zero


def vee(X):
    v = ca.SX(6, 1)
    v[0, 0] = X[2, 1]
    v[1, 0] = X[0, 2]
    v[2, 0] = X[1, 0]
    v[3, 0] = X[3, 0]
    v[4, 0] = X[3, 1]
    v[5, 0] = X[3, 2]
    return v


def wedge(v):
    X = ca.SX.zeros(4, 4)
    X[0, 1] = -v[2]
    X[0, 2] = v[1]
    X[1, 0] = v[2]
    X[1, 2] = -v[0]
    X[2, 0] = -v[1]
    X[2, 1] = v[0]
    X[3, 0] = v[3]
    X[3, 1] = v[4]
    X[3, 2] = v[5]
    return X


class SE3Dcm:

    group_shape = (4, 4)
    group_params = 12
    algebra_params = 6

    def __init__(self):
        raise RuntimeError('this class is just for scoping, do not instantiate')

    @classmethod
    def check_group_shape(cls, a):
        assert a.shape == cls.group_shape or a.shape == (cls.group_shape[0],)

    @classmethod
    def product(cls, a, b):
        cls.check_group_shape(a)
        cls.check_group_shape(b)
        return ca.mtimes(a, b)

    @classmethod
    def inv(cls, a):
        # TODO
        cls.check_group_shape(a)
        return ca.transpose(a)

    @classmethod
    def exp(cls, v):
        # TODO
        theta = ca.norm_2(v)
        X = wedge(v)
        return ca.SX.eye(3) + cls.C1(theta)*X + cls.C2(theta)*ca.mtimes(X, X)

    @classmethod
    def log(cls, R):
        # TODO
        theta = ca.arccos((ca.trace(R) - 1) / 2)
        return vee(cls.C3(theta) * (R - R.T))

    @classmethod
    def kinematics(cls, R, w):
        # TODO
        assert R.shape == (3, 3)
        assert w.shape == (3, 1)
        return ca.mtimes(R, wedge(w))