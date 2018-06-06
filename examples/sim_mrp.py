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
f_shadow = ca.Function('shadow', [a], [a.shadow()])
f_quat = ca.Function('to_quat', [a], [a.to_quat()])
f = lambda t, x: fun(x, [1, -2, 3])

tf = 5

hist = {
    't': [],
    'x': [],
    'q': []
}

t = 0
dt = 0.05
x = np.array([0, 0, 0])
shadow = 0 # need to track shadow state to give a consistent quaternion
while t + dt < tf:
    res = scipy.integrate.solve_ivp(
        fun=f, t_span=[t, t + dt],
        y0=x)
    x = np.array(res['y'][:, -1])
    hist['t'].append(res['t'][-1])
    hist['x'].append(x)
    if np.linalg.norm(x) > 1:
        x = np.reshape(f_shadow(x), -1)
        shadow = not shadow
    q = f_quat(x)
    if shadow:
        q = -q
    hist['q'].append(np.reshape(q, -1))
    t += dt

for k in hist.keys():
    hist[k] = np.array(hist[k])

plt.figure()

plt.subplot(211)
plt.plot(hist['t'], hist['x'])
plt.xlabel('t')
plt.ylabel('mrp')
plt.gca().set_ylim(-1, 1)

plt.subplot(212)
plt.plot(hist['t'], hist['q'])
plt.xlabel('t')
plt.ylabel('q')
plt.gca().set_ylim(-1, 1)

plt.show()
