import casadi as ca
from . import quat, dcm, mrp, euler


def vee(X):
    v = ca.SX(3, 1)
    v[0, 0] = X[2, 1]
    v[1, 0] = X[0, 2]
    v[2, 0] = X[1, 0]
    return v


def wedge(v):
    X = ca.SX(3, 3)
    X[0, 1] = -v[2]
    X[0, 2] = v[1]
    X[1, 0] = v[2]
    X[1, 2] = -v[0]
    X[2, 0] = -v[1]
    X[2, 1] = v[0]
    return X
