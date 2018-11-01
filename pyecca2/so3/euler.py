import casadi as ca
from . import quat


def from_quat(q):
    assert q.shape == (4, 1)
    e = ca.SX(3, 1)
    a = q[0]
    b = q[1]
    c = q[2]
    d = q[3]
    e[0] = ca.atan2(2 * (a * b + c * d), 1 - 2 * (b ** 2 + c ** 2))
    e[1] = ca.asin(2 * (a * c - d * b))
    e[2] = ca.atan2(2 * (a * d + b * c), 1 - 2 * (c ** 2 + d ** 2))
    return e


def from_dcm(R):
    assert R.shape == (3, 3)
    return from_quat(quat.from_dcm(R))


def from_mrp(a):
    assert a.shape == (3, 1)
    return from_quat(quat.from_mrp(a))