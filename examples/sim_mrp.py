"""
Attitude Kalman Filter using Modified Rodrigues Parameters,
Lie group correction ideas from LG-EKF, and invariance ideas from IEKF.
"""
from pyecca.so3.mrp import Mrp
from pyecca.so3.quat import Quat

import casadi as ca
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate


def measurement():
    a = Mrp(ca.SX.sym('a', 3, 1))
    xh = Mrp(ca.SX.sym('xh', 3, 1))
    x = Mrp(ca.SX.sym('x', 3, 1))
    eta_L = (x.inv()*xh)
    eta_L = ca.if_else(ca.norm_2(eta_L) > 1, eta_L.shadow(), eta_L)
    eta_L = Mrp(eta_L).log()

    eta_R = (xh*x.inv())
    eta_R = ca.if_else(ca.norm_2(eta_R) > 1, eta_R.shadow(), eta_R)
    eta_R = Mrp(eta_R).log()

    q = Quat(ca.SX.sym('q', 4, 1))
    w = ca.SX.sym('w', 3, 1)
    fun = ca.Function("dyn", [a, w], [a.derivative(w).T])
    return {
        'left_correct': ca.Function('f_right_correct', [x, w], [Mrp.exp(ca.mtimes(x.to_dcm(), w))*x]),
        'right_correct': ca.Function('f_left_correct', [x, w], [x*Mrp.exp(w)]),
        'eta_L': ca.Function('eta_L', [x, xh], [eta_L]),
        'eta_R': ca.Function('eta_R', [x, xh], [eta_R]),
        'mrp_shadow': ca.Function('mrp_shadow', [a], [a.shadow()]),
        'mrp_to_quat': ca.Function('mrp_to_quat', [a], [a.to_quat()]),
        'quat_to_euler': ca.Function('quat_to_euler', [q], [q.to_euler()]),
        'dynamics': lambda u: lambda t, x: fun(x, u),
        'measure_g': ca.Function('measure_g', [a], [a.to_dcm().T*ca.SX([0, 0, 1])]),
        'measure_hdg': ca.Function('measure_hdg', [a], [a.to_dcm().T*ca.SX([1, 0, 0])]),
    }


def alignment():
    y = ca.SX.sym('y', 3, 1)
    yh = ca.SX.sym('yh', 3, 1)
    c_vect = ca.cross(y, yh)/ca.norm_2(y)/ca.norm_2(yh)
    n_cvect = ca.norm_2(c_vect)
    omega_c = ca.if_else(n_cvect > 0, ca.asin(n_cvect)*c_vect/n_cvect, ca.SX([0, 0, 0]))
    f_correct_align = ca.Function('correct_align', [y, yh], [omega_c])
    return {
        'correct_align': f_correct_align
    }


def jacobians():
    eta_R = ca.SX.sym('eta_R', 6, 1)  # type: ca.SX
    xh = ca.SX.sym('x_h', 6, 1)  # type: ca.SX
    rh = Mrp.exp(xh[0:3])
    re = Mrp.exp(eta_R[0:3])
    be = eta_R[3:6]
    dre = re.derivative(-ca.mtimes(re.to_dcm(), be))
    f = ca.Function('f', [eta_R, xh], [ca.vertcat(dre, ca.SX.zeros(3))])
    F = ca.Function('F', [eta_R, xh], [ca.jacobian(f(eta_R, xh), eta_R)])
    Q = ca.SX.sym('Q', ca.Sparsity.diag(6))  # type: ca.SX
    P0 = ca.SX.sym('P0', ca.Sparsity.diag(6))  # type: ca.SX
    # find sparsity pattern of P
    dP0 = ca.mtimes(F(eta_R, xh), P0) + ca.mtimes(P0, F(eta_R, xh).T) + Q  # type: ca.SX
    PU = ca.SX.sym('P', ca.triu(dP0).sparsity())  # type: ca.SX
    P = ca.triu2symm(PU)  # type: ca.SX
    dP = ca.mtimes(F(eta_R, xh), P) + ca.mtimes(P, F(eta_R, xh).T) + Q  # type: ca.SX
    dP = ca.substitute(dP, eta_R, ca.SX.zeros(6))  # type: ca.SX
    f_dP = ca.Function('dP', [xh, PU, Q], [dP])
    return {
        'f': f,
        'F': F,
        'dP': f_dP
    }


def derive_functions():
    funcs = {}
    funcs.update(measurement())
    funcs.update(alignment())
    funcs.update(jacobians())
    return funcs


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
    'eta_L': [],
    'eta_R': [],
    'std': [],
}

t = 0
tf = 5
dt = 0.005

accel_sqrt_N = 0.005
mag_sqrt_N = 0.005

freq = 0.1
mod_accel = 10
mod_mag = 10

dt_mag = dt*mod_mag
dt_accel = dt*mod_accel

# initial covariance matrix
P = np.diag([10.0, 10.0, 10.0, 1.0, 1.0, 1.0])**2

# process noise
Q_sqrt_N = np.diag([0.001, 0.001, 0.001, 0, 0, 0])
Q = Q_sqrt_N**2/dt

R_accel = np.eye(2)*accel_sqrt_N**2/dt_accel
R_mag = mag_sqrt_N**2/dt_mag

# initial conditions
x = np.array([0.1, 0.2, 0.3])
bg = 1*np.array([-0.5, 0, 0.5])

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
eta_L = np.array([0, 0, 0])
eta_R = np.array([0, 0, 0])


shadow = 0 # need to track shadow state to give a consistent quaternion
shadowh = 0 # need to track shadow state to give a consistent quaternion

i = 0


def handle_shadow(x, s, q):
    if np.linalg.norm(x) > 1:
        x = np.reshape(func['mrp_shadow'](x), -1)
        s = not s
    q = func['mrp_to_quat'](x)
    if s:
        q *= -1
    return x, s, q

H_mag = np.array([
    [0, 0, 1, 0, 0 ,0]
])

H_accel = np.array([
    [1, 0, 0, 0, 0, 0],
    [0 ,1, 0, 0, 0, 0]
])


while t + dt < tf:
    i += 1

    omega = 1*np.array([0.1, 0.2, 0.3]) + 1*np.cos(2*np.pi*freq*t)

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

    P += np.array(func['dP'](ca.vertcat(xh, bgh), P, Q)*dt)

    # correction for accel
    if i % mod_accel == 0:
        y = func['measure_g'](x)
        # the measurement here is not in the angle errors, need to account for this
        yh = func['measure_g'](xh) + np.sqrt(R_accel[0, 0])*np.random.randn(3)
        S = H_accel.dot(P).dot(H_accel.T) + R_accel
        K = P.dot(H_accel.T).dot(np.linalg.inv(S))
        #print('K accel', K[0, 0], K[3, 0], K)
        # TODO, K_accel has two dimensions, same, gain though, how to handle?
        P -= K.dot(H_accel).dot(P)
        omega_c_accel = np.reshape(func['correct_align'](y, yh), -1)*K[0, 0]
        bgh = bgh + K[3, 0]*omega_c_accel
        xh = np.reshape(func['right_correct'](xh, omega_c_accel), -1)
        xh, shadowh, qh = handle_shadow(xh, shadowh, qh)


    # correction for mag
    if i % mod_mag == 0:
        y2 = func['measure_hdg'](x) + np.sqrt(R_mag)*np.random.randn(3)
        y2h = func['measure_hdg'](xh)
        S = H_mag.dot(P).dot(H_mag.T) + R_mag
        K = P.dot(H_mag.T).dot(np.linalg.inv(S))
        P -= K.dot(H_mag).dot(P)
        #print('K mag', K[2, 0], K[5, 0])
        omega_c_mag = np.reshape(func['correct_align'](y2, y2h), -1)*K[2, 0]
        bgh = bgh + K[5, 0]*omega_c_mag
        xh = np.reshape(func['right_correct'](xh, omega_c_mag), -1)
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
    hist['eta_L'].append(np.reshape(func['eta_L'](x, xh), -1))
    hist['eta_R'].append(np.reshape(func['eta_R'](x, xh), -1))
    hist['std'].append(np.sqrt(np.reshape(np.diag(P), -1)))
    t += dt

for k in hist.keys():
    hist[k] = np.array(hist[k])

plt.figure()

plt.subplot(311)
plt.plot(hist['t'], hist['x'], 'r')
plt.plot(hist['t'], hist['xh'], 'k')
plt.grid()
plt.xlabel('t, sec')
plt.ylabel('mrp')
plt.gca().set_ylim(-1, 1)
plt.gca().set_xlim(0, tf)


plt.subplot(312)
plt.plot(hist['t'], hist['q'], 'r')
plt.plot(hist['t'], hist['qh'], 'k')
plt.xlabel('t, sec')
plt.ylabel('q')
plt.grid()
plt.gca().set_ylim(-1, 1)
plt.gca().set_xlim(0, tf)

plt.subplot(313)
plt.plot(hist['t'], np.rad2deg(hist['euler']), 'r')
plt.plot(hist['t'], np.rad2deg(hist['eulerh']), 'k')
plt.xlabel('t, sec')
plt.ylabel('euler, deg')
plt.grid()
plt.gca().set_ylim(-200, 200)
plt.gca().set_xlim(0, tf)

plt.figure()
plt.subplot(311)
plt.plot(hist['t'], hist['y'], 'r')
plt.plot(hist['t'], hist['yh'], 'k')
plt.xlabel('t, sec')
plt.ylabel('y')
plt.gca().set_xlim(0, tf)
plt.grid()
plt.title('accel measurement')

plt.subplot(312)
plt.plot(hist['t'], hist['y2'], 'r')
plt.plot(hist['t'], hist['y2h'], 'k')
plt.xlabel('t, sec')
plt.ylabel('y2')
plt.gca().set_xlim(0, tf)
plt.grid()
plt.title('heading measurement')

plt.subplot(313)
plt.plot(hist['t'], hist['shadow'], 'r')
plt.plot(hist['t'], hist['shadowh'], 'k')
plt.xlabel('t, sec')
plt.ylabel('y2')
plt.gca().set_xlim(0, tf)
plt.grid()
plt.title('shadow')


plt.figure()

#plt.subplot(311)
h_bg = plt.plot(hist['t'], np.rad2deg(hist['bg']), 'r')
std_b =  hist['std'][:, 3:6]
h_sig = plt.plot(hist['t'], np.rad2deg(bg + 3*std_b), 'g')
plt.plot(hist['t'], np.rad2deg(bg -3*std_b), 'g')
h_bgh = plt.plot(hist['t'], np.rad2deg(hist['bgh']), 'k')
plt.xlabel('t, sec')
plt.ylabel('bias, deg/s')
plt.gca().set_xlim(0, tf)
plt.title('gyro bias')
plt.legend([h_bg[0], h_bgh[0], h_sig[0]], ['$b$', '$\hat{b}$', '3 $\sigma$'])
plt.grid()

#plt.subplot(312)
#plt.plot(hist['t'], hist['ba'], 'r--')
#plt.plot(hist['t'], hist['bah'], 'k')
#plt.xlabel('t, sec')
#plt.ylabel('y')
#plt.gca().set_xlim(0, tf)
#plt.title('accel bias')

plt.figure()
std_r =  hist['std'][:, 0:3]
#plt.plot(hist['t'], hist['eta_L'], 'r', label='$\eta_L$', alpha=0.3)
h_sig = plt.plot(hist['t'], np.rad2deg(3*std_r), 'g')
plt.plot(hist['t'], np.rad2deg(-3*std_r), 'g')
h_eta = plt.plot(hist['t'], np.rad2deg(hist['eta_R']), 'k')
plt.gca().set_ylim(-20, 20)
plt.ylabel('error, deg')
plt.legend([h_eta[0], h_sig[0]], ['$\eta$', '3 $\sigma$'])
plt.gca().set_xlim(0, tf)
plt.grid()

plt.show()
