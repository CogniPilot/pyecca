import casadi as ca
from  .. import so3
from . import quat


eps = 1e-8 # tolerance for avoiding divide by 0
SHAPE = (3, 3)

x = ca.SX.sym('x')
C1 = ca.Function('a', [x], [ca.if_else(x < eps, 1 - x ** 2 / 6 + x ** 4 / 120, ca.sin(x)/x)])
C2 = ca.Function('b', [x], [ca.if_else(x < eps, 0.5 - x ** 2 / 24 + x ** 4 / 720, (1 - ca.cos(x)) / x ** 2)])
C3 = ca.Function('d', [x], [ca.if_else(x < eps, 0.5 + x**2/12 + 7*x**4/720, x/(2*ca.sin(x)))])


def product(a, b):
    assert a.shape == SHAPE
    assert b.shape == SHAPE
    return ca.mtimes(a, b)


def inv(a):
    assert a.shape == SHAPE
    return ca.transpose(a)


def exp(v):
    theta = ca.norm_2(v)
    X = so3.wedge(v)
    return ca.SX.eye(3) + C1(theta)*X + C2(theta)*ca.mtimes(X, X)


def log(R):
    theta = ca.arccos((ca.trace(R) - 1) / 2)
    return so3.vee(C3(theta) * (R - R.T))


def derivative(R, w):
    assert R.shape == (3, 3)
    assert w.shape == (3, 1)
    return ca.mtimes(R, so3.wedge(w))


def from_quat(q):
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


def from_mrp(a):
    assert a.shape == (3, 1)
    X = so3.wedge(a)
    n_sq = ca.dot(a, a)
    X_sq = ca.mtimes(X, X)
    R = ca.SX.eye(3) + (8 * X_sq - 4 * (1 - n_sq) * X) / (1 + n_sq) ** 2
    # return transpose, due to convention difference in book
    return R.T


def from_euler(e):
    return from_quat(quat.from_euler(e))