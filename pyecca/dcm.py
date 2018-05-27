"""
A module for Direction Cosine Matrices (DCMs)
"""
import casadi as ca

Expr = ca.SX


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
