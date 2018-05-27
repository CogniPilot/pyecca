"""
A module for Body 321 Euler angles.

This is a representation of SO(3). There are 3 parameters (phi, theta, psi), and it is singular at theta = +/- pi/2.
"""

import casadi as ca

Expr = ca.SX

# TODO