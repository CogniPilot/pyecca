import casadi as ca


class R3:

    SHAPE = (3, 1)

    def __init__(self):
        raise RuntimeError('this class is just for scoping, do not instantiate')

    @classmethod
    def identity(cls):
        return ca.DM([0, 0, 0])

    @classmethod
    def product(cls, a, b):
        assert a.shape == cls.SHAPE
        assert b.shape == cls.SHAPE
        return a + b

    @classmethod
    def inv(cls, a):
        assert a.shape == cls.SHAPE
        return -a

    @classmethod
    def exp(cls, v):
        assert v.shape == cls.SHAPE
        return v

    @classmethod
    def log(cls, v):
        assert v.shape == cls.SHAPE
        return v

