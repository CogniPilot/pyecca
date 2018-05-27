"""
A module for Direction Cosine Matrices (DCMs).

This is the standard representation of SO(3). There are 9 parameters and no singularities.
"""
import casadi as ca

Expr = ca.SX


# coefficient functions, with taylor series approx near origin
eps = 0.001
x = Expr.sym('x')
a = ca.Function('a', [x], [ca.if_else(x < eps, 0.5 + x**2/12 + 7*x**4/720, x/(2*ca.sin(x)))])
b = ca.Function('b', [x], [ca.if_else(x < eps, 0.5 - x ** 2 / 24 + x ** 4 / 720, (1 - ca.cos(x)) / x ** 2)])


class Dcm(Expr):

    def __init__(self, *args):
        super().__init__(*args)
        assert self.shape == (3, 3)

    def __add__(self, other: Expr) -> None:
        assert False

    def __sub__(self, other: Expr) -> None:
        assert False

    def __neg__(self):
        assert False

    def __rmul__(self, other: Expr) -> 'Dcm':
        assert other.shape == (1, 1)
        return Dcm(other * Expr(self))

    def __mul__(self, other: Expr) -> 'Dcm':
        return Dcm(ca.mtimes(self, other))

    def inv(self) -> 'Dcm':
        return Dcm(ca.inv(self))
