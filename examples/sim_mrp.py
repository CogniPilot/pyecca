"""
Demonstrates use of Modified Rodrigues Paramaters (MRPs)
for simulation of attitude kinematics.
"""
from pyecca.so3.mrp import Mrp
from pyecca.so3.quat import Quat

import casadi as ca
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate


def derive_functions():
    a = Mrp(ca.SX.sym('a', 3, 1))
    q = Quat(ca.SX.sym('q', 4, 1))
    w = ca.SX.sym('w', 3, 1)
    fun = ca.Function("dyn", [a, w], [a.derivative(w).T])
    f_mrp_shadow = ca.Function('mrp_shadow', [a], [a.shadow()])
    f_mrp_to_quat = ca.Function('mrp_to_quat', [a], [a.to_quat()])
    f_quat_to_euler = ca.Function('quat_to_euler', [q], [q.to_euler()])
    f_dynamics = lambda t, x: fun(x, [1, -3, 2])
    return {
        'mrp_shadow': f_mrp_shadow,
        'mrp_to_quat': f_mrp_to_quat,
        'quat_to_euler': f_quat_to_euler,
        'dynamics': f_dynamics
    }


func = derive_functions()


tf = 5

hist = {
    't': [],
    'x': [],
    'q': [],
    'euler': [],
}

t = 0
dt = 0.05
x = np.array([0, 0, 0])
shadow = 0 # need to track shadow state to give a consistent quaternion
while t + dt < tf:
    # simulation
    res = scipy.integrate.solve_ivp(
        fun=func['dynamics'], t_span=[t, t + dt],
        y0=x)
    x = np.array(res['y'][:, -1])
    hist['t'].append(res['t'][-1])
    hist['x'].append(x)
    if np.linalg.norm(x) > 1:
        x = np.reshape(func['mrp_shadow'](x), -1)
        shadow = not shadow
    q = func['mrp_to_quat'](x)
    if shadow:
        q = -q
    hist['q'].append(np.reshape(q, -1))
    hist['euler'].append(np.reshape(func['quat_to_euler'](q), -1))
    t += dt

for k in hist.keys():
    hist[k] = np.array(hist[k])

plt.figure()

plt.subplot(311)
plt.plot(hist['t'], hist['x'])
plt.xlabel('t, sec')
plt.ylabel('mrp')
plt.gca().set_ylim(-1, 1)
plt.gca().set_xlim(0, tf)


plt.subplot(312)
plt.plot(hist['t'], hist['q'])
plt.xlabel('t, sec')
plt.ylabel('q')
plt.gca().set_ylim(-1, 1)
plt.gca().set_xlim(0, tf)

plt.subplot(313)
plt.plot(hist['t'], np.rad2deg(hist['euler']))
plt.xlabel('t, sec')
plt.ylabel('euler, deg')
plt.gca().set_ylim(-200, 200)
plt.gca().set_xlim(0, tf)

plt.show()
