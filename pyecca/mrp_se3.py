"""
A module for SE3 Lie Group using Modified Rodriguez Parameters
to parametrize SO3.
"""
import casadi as ca

from .mrp import Mrp


def exp(x: ca.SX) -> ca.SX:
    w = x[:3]
    a = Mrp.exp(w)
    return a
