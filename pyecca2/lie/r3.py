import casadi as ca


SHAPE = (3, 1)


def product(a, b):
    assert a.shape == SHAPE
    assert b.shape == SHAPE
    return a + b


def inv(a):
    assert a.shape == SHAPE
    return -a


def exp(v):
    assert v.shape == SHAPE
    return v


def log(v):
    assert v.shape == SHAPE
    return v

