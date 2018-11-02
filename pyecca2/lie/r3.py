import casadi as ca


class R3:

    group_params = 3
    algebra_params = 3

    def __init__(self):
        raise RuntimeError('this class is just for scoping, do not instantiate')

    @classmethod
    def identity(cls):
        return ca.DM([0, 0, 0])

    @classmethod
    def product(cls, a, b):
        assert a.shape[0] == cls.group_params
        assert b.shape[0] == cls.group_params
        return a + b

    @classmethod
    def inv(cls, a):
        assert a.shape[0] == cls.group_params
        return -a

    @classmethod
    def exp(cls, v):
        assert v.shape[0] == cls.algebra_params
        return v

    @classmethod
    def log(cls, a):
        assert a.shape[0] == cls.group_params
        return a

