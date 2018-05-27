"""
Demonstrates code generation of an EKF prediction and correction step.
"""

import os

import casadi as ca
import numpy as np
from pyecca.filters.ekf import Ekf


def f(symtype, x, u):
    dx = symtype.zeros(6)
    dx[0:3] = x[3:6]
    dx[3:6] = u[0:3]
    return dx


def g(symtype, x):
    y = symtype.zeros(3)
    y = x[0:3]
    return y


kf = Ekf(ca.SX, n_x=6, n_u=3)

print(kf.predict('ekf_predict', f)([1, 2, 3, 4, 5, 6], [1, 1, 1], np.eye(6), [1, 1, 1, 1, 1, 1]))
print(kf.correct('ekf_correct', g)([1, 2, 3, 4, 5, 6], [1, 1, 1], 1*np.eye(6), [0.01, 0.01, 0.01]))

gen = ca.CodeGenerator('casadi_ekf.c', {'main': False, 'mex': False, 'with_header': True})
gen.add(kf.predict('ekf_predict', f))
gen.add(kf.correct('ekf_correct', g))

gen_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'gen/')
if not os.path.exists(gen_dir):
    os.mkdir(gen_dir)
gen.generate(gen_dir)