"""
The script demonstrates generating a c function from a Casadi expression.
"""
from pyecca.so3.mrp import Mrp
import casadi as ca
import os

a = Mrp(ca.SX.sym('a', 3))
omega = ca.SX.sym('omega', 3)

f = ca.Function('mrp_deriv', [a, omega], [a.derivative(omega)], ['a', 'omega'], ['a_dot'])

print('test', f([1, 2, 3], [1, 2, 3]))

gen = ca.CodeGenerator('mrp_deriv.c', {'main': True, 'mex': False, 'with_header': True})
gen.add(f)

gen_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'gen/')
if not os.path.exists(gen_dir):
    os.mkdir(gen_dir)
gen.generate(gen_dir)