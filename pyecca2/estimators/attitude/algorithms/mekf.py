"""
multiplicative kalman filter with quaternions
:return: dict of equations
"""
from .common import *
from . import quat

# x, state (7)
# -----------
# q, quaternion (4)
# b, gyro bias (3)
x = ca.SX.sym('x', 7)
q = x[:4]
b_gyro = x[4:7]
r = so3.Mrp.from_quat(q)
C_nb = so3.Dcm.from_quat(q)
G = DirectProduct([so3.Quat, r3.R3])

# e, error state (6)
# ----------------
# er, so(3) lie algebra rotation error
# eb, R(3) lie algebra rotation error
n_e = 6
eta = ca.SX.sym('eta', n_e, 1)  # (right)
eta_r = eta[0:3]
eta_b = eta[3:6]
W = ca.SX.sym('W', ca.Sparsity_lower(n_e))


def get_state(**kwargs):
    return ca.Function('get_state', [x], [q, r, b_gyro], ['x'], ['q', 'r', 'b_gyro'])


def initialize(**kwargs):
    return quat.initialize(**kwargs)


def predict(**kwargs):

    # state derivative
    xdot = ca.vertcat(so3.Quat.kinematics(q, omega_m - b_gyro + w_gyro), w_gyro_rw)
    f_xdot = ca.Function('xdot', [t, x, omega_m, w_gyro, w_gyro_rw],
                         [xdot], ['t', 'x', 'omega_m', 'w_gyro', 'w_gyro_rw'], ['xdot'])

    # state prop w/o noise
    x1 = util.rk4(lambda t, x: f_xdot(t, x, omega_m, ca.DM.zeros(3), ca.DM.zeros(3)), t, x, dt)

    # normalize quaternion
    n_q1 = ca.norm_2(x1[:4])
    x1[0:4] = ca.if_else(ca.fabs(n_q1 - 1) > 2e-7, x1[:4] / n_q1, x1[:4])

    # error dynamics
    eta_R = so3.Dcm.exp(eta_r)
    f = ca.Function('f', [omega_m, eta, x, w_gyro_rw], [
        ca.vertcat(-ca.mtimes(ca.DM.eye(3) - eta_R, omega_m - b_gyro) - ca.mtimes(eta_R, eta_b),
                   w_gyro_rw)])

    # linearized error dynamics
    F = ca.sparsify(ca.substitute(ca.jacobian(
        f(omega_m, eta, x, w_gyro_rw), eta), eta, ca.SX.zeros(n_e)))

    # covariance propagation
    f_W_dot_lt = ca.Function(
        'W_dot_lt',
        [x, W, std_gyro, sn_gyro_rw, omega_m, dt],
        [ca.tril(util.sqrt_covariance_predict(W, F, Q))])
    W1 = util.rk4(lambda t, y: f_W_dot_lt(x, y, std_gyro, sn_gyro_rw, omega_m, dt), t, W, dt)

    # prediction
    return ca.Function('predict', [t, x, W, omega_m, std_gyro, sn_gyro_rw, dt], [x1, W1],
                          ['t', 'x', 'W', 'omega_m', 'std_gyro', 'sn_gyro_rw', 'dt'], ['x1', 'W1'])


def correct_mag(**kwargs):
    y_b = ca.SX.sym('y_b', 3)
    C_nm = so3.Dcm.exp(mag_decl*e3) # neglect incl

    # meausre yaw in nav frame, but error in body frame for mekf
    dR = so3.Dcm.from_quat(G.exp(eta)[:4])
    B_n = ca.mtimes(so3.Dcm.exp(mag_decl*e3), ca.SX([1, 0, 0]))
    yh_b = ca.mtimes([dR.T, C_nb.T, B_n])

    #H_mag = ca.jacobian(yh_b, eta)
    H_mag = ca.sparsify(
        -ca.horzcat(so3.wedge(ca.mtimes(C_nb.T, B_n)), ca.SX.zeros(3, 3)))

    Rs_mag = std_mag*ca.diag([1, 1, 1])

    W_mag, K_mag, Ss_mag = util.sqrt_correct(Rs_mag, H_mag, W)
    S_mag = ca.mtimes(Ss_mag, Ss_mag.T)
    r_mag = 0*(y_b - ca.substitute(yh_b, eta, ca.SX.zeros(6)))
    x_mag = G.product(x, G.exp(ca.mtimes(K_mag, r_mag)))
    beta_mag = ca.mtimes([r_mag.T, ca.inv(S_mag), r_mag]) / beta_mag_c
    r_std_mag = ca.diag(Ss_mag)

    # ignore correction when near singular point
    mag_ret = 0
    x_mag = ca.if_else(mag_ret == 0, x_mag, x)
    W_mag = ca.if_else(mag_ret == 0, W_mag, W)

    #TODO skipping for now
    x_mag = x
    W_mag = W

    return ca.Function(
        'correct_mag',
        [x, W, y_b, mag_decl, std_mag, beta_mag_c],
        [x_mag, W_mag, beta_mag, r_mag, r_std_mag, mag_ret],
        ['x', 'W', 'y_b', 'decl', 'std_mag', 'beta_mag_c'],
        ['x_mag', 'W_mag', 'beta_mag', 'r_mag', 'r_std_mag', 'error_code'])


def correct_accel(**kwargs):
    H_accel = ca.SX(2, 6)
    H_accel[0, 0] = 1
    H_accel[1, 1] = 1

    f_measure_accel = ca.Function('measure_accel', [x],
        [g*ca.mtimes(C_nb.T, ca.SX([0, 0, -1]))], ['x'],
                                  ['y'])
    yh_accel = f_measure_accel(x)
    y_b = ca.SX.sym('y_b', 3)
    n3 = ca.SX([0, 0, 1])
    y_n = ca.mtimes(C_nb, -y_b)
    v_n = ca.cross(y_n, n3) / ca.norm_2(y_b) / ca.norm_2(n3)
    norm_v = ca.norm_2(v_n)
    vh_n = v_n / norm_v
    omega_c_accel_n = ca.sparsify(ca.if_else(norm_v > 0, ca.asin(norm_v) * vh_n, ca.SX([0, 0, 0])))

    Rs_accel = ca.SX.eye(2) * (std_accel + ca.norm_2(omega_m) ** 2 * std_accel_omega)

    W_accel, K_accel, Ss_accel = util.sqrt_correct(Rs_accel, H_accel, W)
    S_accel = ca.mtimes(Ss_accel, Ss_accel.T)
    r_accel = omega_c_accel_n[0:2]
    r_std_accel = ca.diag(Ss_accel)
    beta_accel = ca.mtimes([r_accel.T, ca.inv(S_accel), r_accel]) / beta_accel_c
    x_accel = G.product(x, G.exp(ca.mtimes(K_accel, r_accel)))
    x_accel = ca.sparsify(x_accel)

    # ignore correction when near singular point
    accel_ret = ca.if_else(
        ca.fabs(ca.norm_2(y_b) - g) > 1.0,  # accel magnitude not close to g,
        1,
        0
    )

    x_accel = ca.if_else(accel_ret == 0, x_accel, x)
    W_accel = ca.if_else(accel_ret == 0, W_accel, W)

    #TODO skipping for now
    x_mag = x
    W_mag = W

    return ca.Function(
        'correct_accel', [x, W, y_b, g, omega_m, std_accel, std_accel_omega, beta_accel_c],
        [x_accel, W_accel, beta_accel, r_accel, r_std_accel, accel_ret],
        ['x', 'W', 'y_b', 'g', 'omega_b', 'std_accel', 'std_accel_omega', 'beta_accel_c'],
        ['x_accel', 'W_accel', 'beta_accel', 'r_accel', 'r_std_accel', 'error_code'])


def constants(**kwargs):
    x0 = ca.DM([1, 0, 0, 0, 0, 0, 0])
    return ca.Function('constants', [], [x0, W0], [], ['x0', 'W0'])


def eqs(**kwargs):
    return {
        'initialize': initialize(**kwargs),
        'predict': predict(**kwargs),
        'correct_mag': correct_mag(**kwargs),
        'correct_accel': correct_accel(**kwargs),
        'get_state': get_state(**kwargs),
        'constants': constants(**kwargs),
    }