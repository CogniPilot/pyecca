"""
For an standard EKF demonstrates:
1. Derivation
2. Simulation
3. Code Generation
"""

import os
import matplotlib.pyplot as plt

import casadi as ca
import numpy as np
import scipy.integrate
from pyecca.filters.ekf import Ekf

# set random seed to fix plots
np.random.seed(1)


def f(expr, x, u):
    """
    prediction function
    :param expr: Casadi expression type
    :param x: state
    :param u: input
    :return: state derivative
    """
    dx = expr.zeros(6)
    v = x[3:6]
    dx[0:3] = v
    dx[3:6] = u[0:3] - v*ca.norm_2(v)
    return dx


def g(expr, x):
    """
    measurement function
    :param expr: Casadi expression type
    :param x: state
    :return: measurement vector
    """
    return x[0:3]


# derive Kalman filter based on functions
n_x = 6
n_u = 3
n_y = 3
kf = Ekf(ca.SX, n_x=n_x, n_u=n_u)
x = ca.SX.sym('x', n_x, 1)
f_measure = ca.Function('measure', [x], [g(ca.SX, x)])
f_state_derivative = kf.state_derivative('ekf_state_derivative', f)
f_covariance_derivative = kf.covariance_derivative('ekf_covariance_derivative', f)
f_correct = kf.correct('ekf_correct', g)

# simulation
tf = 9
hist = {
    't': [],
    'u': [],
    'y': [],
    'x': [],
    'xh': [],
    'P': []
}
t = 0
dt = 0.1
dt_measure = 1
t_last_measure = t

# setup an approximately circular trajectory
r = 10
freq = 0.1
theta_dot = freq * 2 * np.pi
v = r*theta_dot

x = np.array([r, 0, 0, 0, v, 0])
xh = x + 0.1*r*np.random.randn(n_x)
y = np.array([0, 0, 0])
P = np.reshape(np.eye(n_x), -1)
sigma_w = 0.1*np.ones(n_x)
sigma_v = 0.1*np.ones(n_y)

while t + dt < tf:

    theta = theta_dot*t
    u = -r*theta_dot**2*np.array([np.cos(theta), np.sin(theta), 0])

    def f_state(t, x):
        return np.reshape(f_state_derivative(x, u), -1)

    def f_cov(t, P):
        return np.reshape(f_covariance_derivative(x, u, ca.reshape(P, (6, 6)), sigma_w), -1)

    res_x = scipy.integrate.solve_ivp(
        fun=f_state, t_span=[t, t + dt],
        y0=x)
    res_xh = scipy.integrate.solve_ivp(
        fun=f_state, t_span=[t, t + dt],
        y0=xh)
    res_P = scipy.integrate.solve_ivp(
        fun=f_cov, t_span=[t, t + dt],
        y0=P.reshape(-1))
    x = np.array(res_x['y'][:, -1])
    xh = np.array(res_xh['y'][:, -1])
    P = res_P['y'][:, -1]

    if (t - t_last_measure) >= dt_measure:
        t_last_measure = t
        y = f_measure(x) + np.random.randn(n_y)*sigma_v/dt_measure
        y = np.reshape(y, -1)
        xh, P = f_correct(xh, y, np.reshape(P, (6, 6)), sigma_v)
        xh = np.reshape(xh, -1)
        P = np.reshape(P, -1)

    t += dt
    hist['t'].append(t)
    hist['u'].append(u)
    hist['y'].append(y)
    hist['x'].append(x)
    hist['xh'].append(xh)
    hist['P'].append(P)

for k in hist.keys():
    hist[k] = np.array(hist[k])

plt.figure()

plt.subplot(3, 1, 1)
plt.plot(hist['t'], hist['x'], 'k')
plt.plot(hist['t'], hist['xh'], 'r-')
plt.xlabel('t')
plt.ylabel('x')

plt.subplot(3, 1, 2)
plt.plot(hist['t'], hist['P'])
plt.xlabel('t')
plt.ylabel('P')

plt.subplot(3, 1, 3)
e = hist['xh'] - hist['x']
plt.plot(hist['t'], e)
plt.xlabel('t')
plt.ylabel('e')

plt.figure()

plt.subplot(3, 1, 1)
plt.plot(hist['x'][:, 0], hist['x'][:, 1], 'k')
plt.plot(hist['xh'][:, 0], hist['xh'][:, 1], 'r-')
plt.xlabel('x_1')
plt.ylabel('x_2')

plt.subplot(3, 1, 2)
plt.plot(hist['t'], hist['u'])
plt.xlabel('t')
plt.ylabel('u')

plt.subplot(3, 1, 3)
plt.plot(hist['t'], hist['y'])
plt.xlabel('t')
plt.ylabel('y')

plt.show()


# code generation
gen = ca.CodeGenerator('casadi_ekf.c', {'main': False, 'mex': False, 'with_header': True})
gen.add(kf.state_derivative('ekf_state_derivative', f))
gen.add(kf.state_derivative('ekf_covariance_derivative', f))
gen.add(kf.correct('ekf_correct', g))
gen_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'gen/')
if not os.path.exists(gen_dir):
    os.mkdir(gen_dir)
gen.generate(gen_dir)
