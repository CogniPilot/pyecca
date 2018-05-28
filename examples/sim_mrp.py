"""
Demonstrates use of Modified Rodrigues Paramaters (MRPs)
for simulation of attitude kinematics.
"""
from pyecca.so3.mrp import Mrp
import casadi as ca
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate

a = Mrp(ca.SX.sym('a', 3, 1))
w = ca.SX.sym('w', 3, 1)
fun = ca.Function("dyn", [a, w], [a.derivative(w).T])
f_shadow = ca.Function('shadow', [a], [Mrp(a).shadow()])
f = lambda t, x: fun(x, [1, -2, 3])

tf = 5

hist = {
    't': [],
    'x': []
}

t = 0
dt = 0.1
x = np.array([0, 0, 0])
while t + dt < tf:
    res = scipy.integrate.solve_ivp(
        fun=f, t_span=[t, t + dt],
        y0=x)
    x = np.array(res['y'][:, -1])
    hist['t'].append(res['t'][-1])
    hist['x'].append(x)
    if np.linalg.norm(x) > 1:
        x = np.reshape(f_shadow(x), -1)
    t += dt

for k in hist.keys():
    hist[k] = np.array(hist[k])

plt.plot(hist['t'], hist['x'])
plt.xlabel('t')
plt.ylabel('x')
plt.show()
