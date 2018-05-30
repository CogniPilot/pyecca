"""
For an standard EKF demonstrates:
1. Derivation
2. Simulation
3. Code Generation
"""

import os
import matplotlib.pyplot as plt
from enum import Enum

import casadi as ca
import numpy as np
import scipy.integrate
from pyecca.filters.ekf import Ekf
from pyecca.so3.mrp import Mrp

# set random seed to fix plots
np.random.seed(1)


class State:

    n = 15

    def __init__(self, x):
        self.x = x
        self.r = Mrp(x[0:3])
        self.b_g = x[3:6]
        self.p_N = x[6:9]
        self.v_N = x[9:12]
        self.b_a = x[12:15]


class Input:

    n = 6

    def __init__(self, u):
        self.u = u
        self.omega_B = u[0:3]
        self.a_B = u[3:6]


def f(expr, x, u):
    """
    dynamics function, a simple acceleration input with added drag
    proportional to the speed^2
    :param expr: Casadi expression type
    :param x: state
    :param u: input
    :return: state derivative
    """
    X = State(x)
    U = Input(u)
    g = 9.8
    dx = State(expr.zeros(15))
    dx.r = X.r.derivative(u.omega_B)
    dx.b_g = expr.zeros(3)
    dx.p_N = X.v_N
    dx.v_N = X.r.to_dcm()*U.a_B - g
    dx.b_a = expr.zeros(3)
    return dx


def g_gps(expr, x):
    """
    measurement function
    :param expr: Casadi expression type
    :param x: state
    :return: measurement vector
    """
    X = State(x)
    return ca.vertcat(X.p_N, X.v_N)


def g_mag(expr, x):
    """
    measurement function
    :param expr: Casadi expression type
    :param x: state
    :return: measurement vector
    """
    X = State(x)
    B_N = ca.SX([1, 0, -1])
    B_B = ca.transpose(X.r.to_dcm())*B_N
    return ca.vertcat(X.p_N, X.v_N)


def derive_ekf(f, g_gps, g_mag):
    # derive filter based on functions
    filt = Ekf(ca.SX, n_x=State.n, n_u=Input.n)
    x = ca.SX.sym('x', State.n, 1)
    f_measure_gps = ca.Function('measure', [x], [g_gps(ca.SX, x)])
    f_measure_mag = ca.Function('measure', [x], [g_mag(ca.SX, x)])
    f_state_derivative = filt.state_derivative('ekf_state_derivative', f)
    f_covariance_derivative = filt.covariance_derivative('ekf_covariance_derivative', f)
    f_correct_gps = filt.correct('ekf_correct_gps', g_gps)
    f_correct_mag = filt.correct('ekf_correct_mag', g_mag)
    return {
        'f': f,
        'measure_gps': f_measure_gps,
        'correct_gps': f_correct_gps,
        'measure_mag': f_measure_mag,
        'correct_mag': f_correct_mag,
        'state_derivative': f_state_derivative,
        'covariance_derivative': f_covariance_derivative
    }


def sim_circle(filter_eqs):

    # derive filter based on functions
    n_x = 6
    n_u = 3
    n_y = 3
    x = ca.SX.sym('x', n_x, 1)
    f_measure = filter_eqs['measure']
    f_state_derivative = filter_eqs['state_derivative']
    f_covariance_derivative = filter_eqs['covariance_derivative']
    f_correct = filter_eqs['correct']

    # simulation
    tf = 9
    data = {
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
        data['t'].append(t)
        data['u'].append(u)
        data['y'].append(y)
        data['x'].append(x)
        data['xh'].append(xh)
        data['P'].append(P)

    for k in data.keys():
        data[k] = np.array(data[k])
    return data


def plot(data):
    plt.figure()

    plt.subplot(3, 1, 1)
    plt.plot(data['t'], data['x'], 'k')
    plt.plot(data['t'], data['xh'], 'r-')
    plt.xlabel('t')
    plt.ylabel('x')

    plt.subplot(3, 1, 2)
    plt.plot(data['t'], data['P'])
    plt.xlabel('t')
    plt.ylabel('P')

    plt.subplot(3, 1, 3)
    e = data['xh'] - data['x']
    plt.plot(data['t'], e)
    plt.xlabel('t')
    plt.ylabel('e')

    plt.figure()

    plt.subplot(3, 1, 1)
    plt.plot(data['x'][:, 0], data['x'][:, 1], 'k')
    plt.plot(data['xh'][:, 0], data['xh'][:, 1], 'r-')
    plt.xlabel('x_1')
    plt.ylabel('x_2')

    plt.subplot(3, 1, 2)
    plt.plot(data['t'], data['u'])
    plt.xlabel('t')
    plt.ylabel('u')

    plt.subplot(3, 1, 3)
    plt.plot(data['t'], data['y'])
    plt.xlabel('t')
    plt.ylabel('y')


def generate_code(eqs):
    # code generation
    gen = ca.CodeGenerator('casadi_ekf.c', {'main': False, 'mex': False, 'with_header': True})
    gen.add(eqs['state_derivative'])
    gen.add(eqs['covariance_derivative'])
    gen.add(eqs['correct'])
    gen_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'gen/')
    if not os.path.exists(gen_dir):
        os.mkdir(gen_dir)
    gen.generate(gen_dir)
    gen.generate('/home/jgoppert/git/phd/px4/src/modules/cei/')


if __name__ == "__main__":
    ekf_eqs = derive_ekf(f, g_gps, g_mag)
    ekf_data = sim_circle(ekf_eqs)
    generate_code(ekf_eqs)
    plot(ekf_data)
    plt.show()
