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
    f_dynamics = lambda u: lambda t, x: fun(x, u)
    f_measure_g = ca.Function('measure_g', [a], [a.to_dcm().T*ca.SX([0, 0, 1])])
    f_measure_hdg = ca.Function('measure_hdg', [a], [a.to_dcm().T*ca.SX([1, 0, 0])])

    # gravity alignment correction
    y = ca.SX.sym('y', 3, 1)
    yh = ca.SX.sym('yh', 3, 1)
    c_vect = ca.cross(y, yh)/ca.norm_2(y)/ca.norm_2(yh)
    n_cvect = ca.norm_2(c_vect)
    omega_c = ca.if_else(n_cvect > 0, ca.asin(n_cvect)*c_vect/n_cvect, ca.SX([0, 0, 0]))
    f_correct_align = ca.Function('correct_align', [a, y, yh], [omega_c])

    # magnetic heading correction

    return {
        'mrp_shadow': f_mrp_shadow,
        'mrp_to_quat': f_mrp_to_quat,
        'quat_to_euler': f_quat_to_euler,
        'dynamics': f_dynamics,
        'measure_g': f_measure_g,
        'measure_hdg': f_measure_hdg,
        'correct_align': f_correct_align
    }


func = derive_functions()

hist = {
    't': [],
    'shadow': [],
    'shadowh': [],
    'x': [],
    'y': [],
    'y2': [],
    'q': [],
    'euler': [],
    'bg': [],
    'ba': [],
    'xh': [],
    'yh': [],
    'y2h': [],
    'qh': [],
    'eulerh': [],
    'bgh': [],
    'bah': [],
}

t = 0
tf = 100
dt = 0.05

x = np.array([0.5, -0.2, 0.3])
bg = np.array([0.1, 0.2, 0.3])
bgh = np.zeros(3)
ba = np.array([0, 0, 0])
bah = np.zeros(3)
q = np.reshape(func['mrp_to_quat'](x), -1)
xh = np.array([0, 0, 0])
qh = np.reshape(func['mrp_to_quat'](xh), -1)
y = np.reshape(func['measure_g'](x), -1)
y2 = np.reshape(func['measure_hdg'](x), -1)
yh = np.array([0, 0, 0])
y2h = np.array([0, 0, 0])


shadow = 0 # need to track shadow state to give a consistent quaternion
shadowh = 0 # need to track shadow state to give a consistent quaternion

omega = [0.1, 0.2, 0.3]
i = 0


def handle_shadow(x, s, q):
    if np.linalg.norm(x) > 1:
        x = np.reshape(func['mrp_shadow'](x), -1)
        s = not s
    q = func['mrp_to_quat'](x)
    if s:
        q *= -1
    return x, s, q


while t + dt < tf:
    i += 1

    # simulation
    res = scipy.integrate.solve_ivp(
        fun=func['dynamics'](omega), t_span=[t, t + dt],
        y0=x)
    x = np.array(res['y'][:, -1])
    x, shadow, q = handle_shadow(x, shadow, q)
    omega_meas = omega + bg

    # prediction
    resh = scipy.integrate.solve_ivp(
        fun=func['dynamics'](omega_meas - bgh), t_span=[t, t + dt],
        y0=xh)
    xh = np.array(resh['y'][:, -1])
    xh, shadowh, qh = handle_shadow(xh, shadowh, qh)

    # correction for accel
    if i % 1 == 0:
        y = func['measure_g'](x)
        yh = func['measure_g'](xh) + 0.1*np.random.randn(3)
        K = 0.5
        Kb = 0.07
        omega_c_accel = np.reshape(func['correct_align'](xh, y, yh), -1)*K
        bgh = bgh - Kb*omega_c_accel
        resh = scipy.integrate.solve_ivp(
            fun=func['dynamics'](omega_c_accel), t_span=[0, 1],
            y0=xh)
        xh = np.array(resh['y'][:, -1])
        xh, shadowh, qh = handle_shadow(xh, shadowh, qh)

    # correction for mag
    if i % 3 == 0:
        y2 = func['measure_hdg'](x) + 0.1*np.random.randn(3)
        y2h = func['measure_hdg'](xh)
        K = 0.2
        Kb = 0.07
        omega_c_mag = np.reshape(func['correct_align'](xh, y2, y2h), -1)*K
        bgh = bgh - Kb*omega_c_mag
        resh = scipy.integrate.solve_ivp(
            fun=func['dynamics'](omega_c_mag), t_span=[0, 1],
            y0=xh)
        xh = np.array(resh['y'][:, -1])
        xh, shadowh, qh = handle_shadow(xh, shadowh, qh)

    # data
    hist['t'].append(res['t'][-1])
    hist['shadow'].append(shadow)
    hist['shadowh'].append(shadowh)
    hist['x'].append(np.reshape(x, -1))
    hist['y'].append(np.reshape(y, -1))
    hist['y2'].append(np.reshape(y2, -1))
    hist['q'].append(np.reshape(q, -1))
    hist['euler'].append(np.reshape(func['quat_to_euler'](q), -1))
    hist['bg'].append(np.reshape(bg, -1))
    hist['ba'].append(np.reshape(ba, -1))
    hist['xh'].append(np.reshape(xh, -1))
    hist['yh'].append(np.reshape(yh, -1))
    hist['y2h'].append(np.reshape(y2h, -1))
    hist['qh'].append(np.reshape(qh, -1))
    hist['bgh'].append(np.reshape(bgh, -1))
    hist['bah'].append(np.reshape(bah, -1))
    hist['eulerh'].append(np.reshape(func['quat_to_euler'](qh), -1))
    t += dt

print(hist['bgh'])

for k in hist.keys():
    hist[k] = np.array(hist[k])

plt.figure()

plt.subplot(311)
plt.plot(hist['t'], hist['x'], 'r--')
plt.plot(hist['t'], hist['xh'], 'k')
plt.xlabel('t, sec')
plt.ylabel('mrp')
plt.gca().set_ylim(-1, 1)
plt.gca().set_xlim(0, tf)


plt.subplot(312)
plt.plot(hist['t'], hist['q'], 'r--')
plt.plot(hist['t'], hist['qh'], 'k')
plt.xlabel('t, sec')
plt.ylabel('q')
plt.gca().set_ylim(-1, 1)
plt.gca().set_xlim(0, tf)

plt.subplot(313)
plt.plot(hist['t'], np.rad2deg(hist['euler']), 'r--')
plt.plot(hist['t'], np.rad2deg(hist['eulerh']), 'k')
plt.xlabel('t, sec')
plt.ylabel('euler, deg')
plt.gca().set_ylim(-200, 200)
plt.gca().set_xlim(0, tf)

plt.figure()
plt.subplot(311)
plt.plot(hist['t'], hist['y'], 'r--')
plt.plot(hist['t'], hist['yh'], 'k')
plt.xlabel('t, sec')
plt.ylabel('y')
plt.gca().set_xlim(0, tf)
plt.title('accel measurement')

plt.subplot(312)
plt.plot(hist['t'], hist['y2'], 'r--')
plt.plot(hist['t'], hist['y2h'], 'k')
plt.xlabel('t, sec')
plt.ylabel('y2')
plt.gca().set_xlim(0, tf)
plt.title('heading measurement')

plt.subplot(313)
plt.plot(hist['t'], hist['shadow'], 'r--')
plt.plot(hist['t'], hist['shadowh'], 'k')
plt.xlabel('t, sec')
plt.ylabel('y2')
plt.gca().set_xlim(0, tf)
plt.title('shadow')


plt.figure()

plt.subplot(311)
print(hist['bgh'])
plt.plot(hist['t'], hist['bg'], 'r--')
plt.plot(hist['t'], hist['bgh'], 'k')
plt.xlabel('t, sec')
plt.ylabel('y')
plt.gca().set_xlim(0, tf)
plt.title('gyro bias')


plt.subplot(312)
plt.plot(hist['t'], hist['ba'], 'r--')
plt.plot(hist['t'], hist['bah'], 'k')
plt.xlabel('t, sec')
plt.ylabel('y')
plt.gca().set_xlim(0, tf)
plt.title('accel bias')


plt.show()
