import casadi as ca
from pyecca2.lie import so3
from . import quat


eps = 1e-8 # tolerance for avoiding divide by 0


def product(r1, r2):
    assert r1.shape == (4, 1) or r1.shape == (4,)
    assert r2.shape == (4, 1) or r2.shape == (4,)
    a = r1[:3]
    b = r2[:3]
    na_sq = ca.dot(a, a)
    nb_sq = ca.dot(b, b)
    res = ca.SX(4, 1)
    res[:3] = ((1 - na_sq) * b + (1 - nb_sq) * a - 2 * ca.cross(b, a)) \
           / (1 + na_sq * nb_sq - 2 * ca.dot(b, a))
    res[3] = 0  # shadow state
    return res


def inv(r):
    assert r.shape == (4, 1) or r.shape == (4,)
    return ca.vertcat(-r[:3], r[3])


def exp(v):
    assert v.shape == (3, 1) or v.shape == (3,)
    angle = ca.norm_2(v)
    res = ca.SX(4, 1)
    res[:3] = ca.if_else(angle < eps, ca.DM([0, 0, 0]), ca.tan(angle / 4) * v / angle)
    res[3] = 0
    return res


def log(r):
    assert r.shape == (4, 1) or r.shape == (4,)
    n = ca.norm_2(r[:3])
    return ca.if_else(n < eps, ca.DM([0, 0, 0]), 4 * ca.atan(n) * r[:3] / n)


def shadow(r):
    assert r.shape == (4, 1) or r.shape == (4,)
    n_sq = ca.dot(r[:3], r[:3])
    res = ca.SX(4, 1)
    res[:3] = ca.if_else(n_sq > eps, -r[:3] / n_sq, [0, 0, 0])
    res[3] = ca.logic_not(r[3])
    return res


def shadow_if_necessary(r):
    assert r.shape == (4, 1) or r.shape == (4,)
    return ca.if_else(ca.norm_2(r[:3]) > 1, shadow(r), r)


def kinematics(r, w):
    assert r.shape == (4, 1) or r.shape == (4,)
    assert w.shape == (3, 1) or w.shape == (3,)
    a = r[:3]
    n_sq = ca.dot(a, a)
    X = so3.wedge(a)
    B = 0.25 * ((1 - n_sq) * ca.SX.eye(3) + 2 * X + 2 * ca.mtimes(a, ca.transpose(a)))
    return ca.vertcat(ca.mtimes(B, w), 0)


def from_quat(q):
    assert q.shape == (4, 1) or q.shape == (4,)
    x = ca.SX(4, 1)
    den = 1 + q[0]
    x[0] = q[1] / den
    x[1] = q[2] / den
    x[2] = q[3] / den
    x[3] = 0
    return x


def from_dcm(R):
    return from_quat(quat.from_dcm(R))


def from_euler(e):
    return from_quat(quat.from_euler(e))
