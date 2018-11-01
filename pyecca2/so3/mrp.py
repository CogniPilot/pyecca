import casadi as ca
from  .. import so3
from . import quat


eps = 1e-8 # tolerance for avoiding divide by 0
SHAPE = (3, 1)


def product(a, b):
    assert a.shape == SHAPE
    assert b.shape == SHAPE
    na_sq = ca.dot(a, a)
    nb_sq = ca.dot(b, b)
    return ((1 - na_sq) * b + (1 - nb_sq) * a - 2 * ca.cross(b, a)) \
           / (1 + na_sq * nb_sq - 2 * ca.dot(b, a))


def inv(a):
    assert a.shape == SHAPE
    return -a


def exp(v):
    assert v.shape == SHAPE
    angle = ca.norm_2(v)
    return ca.if_else(angle < eps, ca.DM([0, 0, 0]), ca.tan(angle / 4) * v / angle)


def log(a):
    assert a.shape == SHAPE
    n = ca.norm_2(a)
    return ca.if_else(n < eps, ca.DM([0, 0, 0]), 4 * ca.atan(n) * a / n)


def shadow(a):
    assert a.shape == SHAPE
    n_sq = ca.dot(a, a)
    return ca.if_else(n_sq > eps, -a / n_sq, [0, 0, 0])


def derivative(a, w):
    assert a.shape == SHAPE
    assert w.shape == (3, 1)
    n_sq = ca.dot(a, a)
    X = so3.wedge(a)
    B = 0.25 * ((1 - n_sq) * ca.SX.eye(3) + 2 * X + 2 * ca.mtimes(a, ca.transpose(a)))
    return ca.mtimes(B, w)


def from_quat(q):
    assert q.shape == (4, 1)
    x = ca.SX(3, 1)
    den = 1 + q[0]
    x[0] = q[1] / den
    x[1] = q[2] / den
    x[2] = q[3] / den
    return x


def from_dcm(R):
    return from_quat(quat.from_dcm(R))


def from_euler(e):
    return from_quat(quat.from_euler(e))
