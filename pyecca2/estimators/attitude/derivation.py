import casadi as ca

import pyecca2.util as util
from pyecca2.lie.so3 import quat, mrp, dcm

"""
This module derives various attitude estimators using casadi.
"""


def derivation():
    """
    This function derives various attitude estimators. The common problem parameters are at the
    top and these variables can be shared, but each estimator has it's own scope.
    :return: dict of various derivations
    """
    # misc variables
    omega_t = ca.SX.sym('omega_t', 3, 1)  # angular velocity in body frame, true
    omega_m = ca.SX.sym('omega_m', 3, 1)  # angular velocity in body frame, measured
    t = ca.SX.sym('t')  # time
    dt = ca.SX.sym('dt')  # integration time step
    std_mag = ca.SX.sym('std_mag')
    std_gyro = ca.SX.sym('std_gyro')
    std_accel = ca.SX.sym('std_accel')
    sn_gyro_rw = ca.SX.sym('sn_gyro_rw')

    # constants
    mag_decl = ca.SX.sym('mag_decl')
    mag_incl = ca.SX.sym('mag_incl')  # only useful for sim, neglected in correction
    mag_str = ca.SX.sym('mag_str')  # mag field strength
    g = ca.SX.sym('g')

    # noise, mean zero, variance 1
    w_mag = ca.SX.sym('w_mag', 3, 1)
    w_gyro = ca.SX.sym('w_gyro', 3, 1)
    w_gyro_rw = ca.SX.sym('w_gyro_rw', 3, 1)
    w_accel = ca.SX.sym('w_accel', 3, 1)

    std_gyro_rw = sn_gyro_rw / ca.sqrt(dt)
    Q = ca.diag(ca.vertcat(std_gyro, std_gyro, std_gyro, std_gyro_rw, std_gyro_rw, std_gyro_rw) ** 2)

    # e1 = ca.SX([1, 0, 0])
    e2 = ca.SX([0, 1, 0])
    e3 = ca.SX([0, 0, 1])

    def sim_derivation():
        """
        The equations required for simulation.
        :return: dict of equations
        """

        # x, state (7)
        # -----------
        # r, mrp (3)
        # s, shadow, mrp shadow state (1)
        # b, gyro bias (3)
        x = ca.SX.sym('x', 7)
        r = x[0:4]  # last state is shadow state
        b_gyro = x[4:7]
        q = quat.from_mrp(r)
        get_state = ca.Function('get_state', [x], [q, b_gyro], ['x'], ['q', 'b_gyro'])

        # state derivative
        xdot = ca.vertcat(mrp.kinematics(r, omega_t), std_gyro_rw * w_gyro_rw)
        f_xdot = ca.Function('xdot', [t, x, omega_t, sn_gyro_rw, w_gyro_rw],
                             [xdot], ['t', 'x', 'omega_t', 'sn_gyro_rw', 'w_gyro_rw'], ['xdot'])

        # state prop with noise
        x1_sim = util.rk4(lambda t, x: f_xdot(t, x, omega_t, sn_gyro_rw, w_gyro_rw), t, x, dt)
        x1_sim[:4] = mrp.shadow_if_necessary(x1_sim[:4])
        simulate = ca.Function('simulate', [t, x, omega_t, sn_gyro_rw,
                                            w_gyro_rw, dt], [x1_sim],
                               ['t', 'x', 'omega_t', 'sn_gyro_rw',
                                'w_gyro_rw', 'dt'], ['x1'])

        # get dcm from mrp
        C_nb = dcm.from_mrp(r)

        # measure gyro
        measure_gyro = ca.Function('measure_gyro', [x, omega_t, std_gyro, w_gyro],
                                   [omega_t + b_gyro + w_gyro * std_gyro],
                                   ['x', 'omega_t', 'std_gyro', 'w_gyro'], ['y'])

        # measure_mag
        C_nm = dcm.product(dcm.exp(mag_decl*e3), dcm.exp(-mag_incl * e2))
        B_n = mag_str * ca.mtimes(C_nm, ca.SX([1, 0, 0]))
        measure_mag = ca.Function(
            'measure_mag', [x, mag_str, mag_decl, mag_incl, std_mag, w_mag],
            [ca.mtimes(C_nb.T, B_n) + w_mag * std_mag],
            ['x', 'mag_str', 'mag_decl', 'mag_incl', 'std_mag', 'w_mag'], ['y'])

        # measure accel
        measure_accel = ca.Function(
            'measure_accel',
            [x, g, std_accel, w_accel],
            [g * ca.mtimes(C_nb.T, ca.SX([0, 0, -1])) + w_accel * std_accel],
            ['x', 'g', 'std_accel', 'w_accel'], ['y'])

        # constants
        x0 = ca.DM([0, 0, 0, 0, 0, 0, 0])
        constants = ca.Function('constants', [], [x0], [], ['x0'])

        # rotation error
        q1 = ca.SX.sym('q1', 4, 1)
        q2 = ca.SX.sym('q2', 4, 1)
        xi = quat.log(quat.product(quat.inv(q1), q2))
        rotation_error = ca.Function('rotation_error', [q1, q2], [xi], ['q1', 'q2'], ['xi'])

        return {
            'simulate': simulate,
            'measure_gyro': measure_gyro,
            'measure_mag': measure_mag,
            'measure_accel': measure_accel,
            'rotation_error': rotation_error,
            'get_state': get_state,
            'constants': constants
        }

    def mrp_derivation():
        """
        A right invariant extended kalman filter parameterized with
        modified rodrigues parameters
        :return: dict of equations
        """

        # x, state (7)
        # -----------
        # r, mrp (3)
        # s, shadow, mrp shadow state (1)
        # b, gyro bias (3)
        x = ca.SX.sym('x', 7)
        r = x[0:4]  # last state is shadow state
        b_gyro = x[4:7]

        # quaternion from mrp
        q = quat.from_mrp(r)
        get_state = ca.Function('get_state', [x], [q, b_gyro], ['x'], ['q', 'b_gyro'])

        # state derivative
        xdot = ca.vertcat(mrp.kinematics(r, omega_m - b_gyro), std_gyro_rw * w_gyro_rw)
        f_xdot = ca.Function('xdot', [t, x, omega_m, std_gyro, sn_gyro_rw, w_gyro, w_gyro_rw],
                             [xdot], ['t', 'x', 'omega_m', 'std_gyro', 'sn_gyro_rw', 'w_gyro', 'w_gyro_rw'], ['xdot'])

        # state prop w/o noise
        x1 = util.rk4(lambda t, x: f_xdot(
            t, x, omega_m, 0, 0, ca.DM.zeros(3), ca.DM.zeros(3)), t, x, dt)
        x1[:4] = mrp.shadow_if_necessary(x1[:4])

        # e, error state (6)
        # ----------------
        # er, so(3) lie algebra rotation error
        # eb, R(3) lie algebra rotation error
        n_e = 6
        eta = ca.SX.sym('eta', n_e, 1)  # (right)

        # error dynamics
        f = ca.Function('f', [omega_m, eta, x, w_gyro_rw], [
            ca.vertcat(-ca.mtimes(dcm.from_mrp(r), eta[3:6]), w_gyro_rw)])

        # linearized error dynamics
        F = ca.sparsify(ca.substitute(ca.jacobian(f(omega_m, eta, x, w_gyro_rw), eta), eta, ca.SX.zeros(n_e)))

        # covariance propagation
        W = ca.SX.sym('W', ca.Sparsity_lower(n_e))
        f_W_dot_lt = ca.Function(
            'W_dot_lt',
            [x, W, std_gyro, sn_gyro_rw, omega_m, dt],
            [ca.tril(util.sqrt_covariance_predict(W, F, Q))])
        W1 = util.rk4(lambda t, y: f_W_dot_lt(x, y, std_gyro, sn_gyro_rw, omega_m, dt), t, W, dt)

        # combined prediction function
        predict = ca.Function('predict', [t, x, W, omega_m, std_gyro, sn_gyro_rw, dt], [x1, W1],
                              ['t', 'x', 'W', 'omega_m', 'std_gyro', 'sn_gyro_rw', 'dt'], ['x1', 'W1'])

        # get dcm from mrp
        C_nb = dcm.from_mrp(r)

        # mag correction
        C_nm = dcm.product(dcm.exp(mag_decl * e3), dcm.exp(-mag_incl * e2))
        B_n = mag_str * ca.mtimes(C_nm, ca.SX([1, 0, 0]))
        yh_mag = ca.mtimes(C_nb.T, B_n)
        y_mag = ca.SX.sym('y_mag', 3, 1)
        H = ca.jacobian(yh_mag, x)
        correct_mag = ca.Function('correct_mag', [x, W, y_mag], [x, 0.9 * W])

        # accel correction
        y_accel = ca.SX.sym('y_accel', 3, 1)
        correct_accel = ca.Function('correct_accel', [x, W, y_accel], [x, 0.9 * W])

        # constants
        x0 = ca.DM.zeros(7)
        W0 = 1e-3 * ca.DM.eye(n_e)
        constants = ca.Function('constants', [], [x0, W0], [], ['x0', 'W0'])

        return {
            'predict': predict,
            'correct_mag': correct_mag,
            'correct_accel': correct_accel,
            'get_state': get_state,
            'constants': constants
        }

    def quat_derivation():
        """
        right invariant kalman filter with quaternions
        :return: dict of equations
        """

        # x, state (7)
        # -----------
        # q, quaternion (4)
        # b, gyro bias (3)
        x = ca.SX.sym('x', 7)
        q = x[:4]
        b_gyro = x[4:7]
        get_state = ca.Function('get_state', [x], [q, b_gyro], ['x'], ['q', 'b_gyro'])

        # state derivative
        xdot = ca.vertcat(quat.kinematics(q, omega_m - b_gyro + w_gyro), w_gyro_rw)
        f_xdot = ca.Function('xdot', [t, x, omega_m, w_gyro, w_gyro_rw],
                             [xdot], ['t', 'x', 'omega_m', 'w_gyro', 'w_gyro_rw'], ['xdot'])

        # state prop w/o noise
        x1 = util.rk4(lambda t, x: f_xdot(t, x, omega_m, ca.DM.zeros(3), ca.DM.zeros(3)), t, x, dt)
        n_q1 = ca.norm_2(x1[:4])

        # normalize quaternion
        x1[0:4] = ca.if_else(ca.fabs(n_q1 - 1) > 1e-6, x1[:4] / n_q1, x1[:4])

        # e, error state (6)
        # ----------------
        # er, so(3) lie algebra rotation error
        # eb, R(3) lie algebra rotation error
        n_e = 6
        eta = ca.SX.sym('eta', n_e, 1)  # (right)

        # error dynamics
        f = ca.Function('f', [omega_m, eta, x, w_gyro_rw], [
            ca.vertcat(-ca.mtimes(dcm.from_quat(q), eta[3:6]), w_gyro_rw)])

        # linearized error dynamics
        F = ca.sparsify(ca.substitute(ca.jacobian(f(omega_m, eta, x, w_gyro_rw), eta), eta, ca.SX.zeros(n_e)))

        # covariance propagation
        W = ca.SX.sym('W', ca.Sparsity_lower(n_e))
        f_W_dot_lt = ca.Function(
            'W_dot_lt',
            [x, W, std_gyro, sn_gyro_rw, omega_m, dt],
            [ca.tril(util.sqrt_covariance_predict(W, F, Q))])
        W1 = util.rk4(lambda t, y: f_W_dot_lt(x, y, std_gyro, sn_gyro_rw, omega_m, dt), t, W, dt)

        # prediction
        predict = ca.Function('predict', [t, x, W, omega_m, std_gyro, sn_gyro_rw, dt], [x1, W1],
                              ['t', 'x', 'W', 'omega_m', 'std_gyro', 'sn_gyro_rw', 'dt'], ['x1', 'W1'])

        # mag correction
        y_mag = ca.SX.sym('y_mag', 3, 1)
        correct_mag = ca.Function('correct_mag', [x, W, y_mag], [x, 0.9 * W])

        # accel correction
        y_accel = ca.SX.sym('y_accel', 3, 1)
        correct_accel = ca.Function('correct_accel', [x, W, y_accel], [x, 0.9 * W])

        # constants
        x0 = ca.DM([1, 0, 0, 0, 0, 0, 0])
        W0 = 1e-3 * ca.DM.eye(n_e)
        constants = ca.Function('constants', [], [x0, W0], [], ['x0', 'W0'])

        return {
            'predict': predict,
            'correct_mag': correct_mag,
            'correct_accel': correct_accel,
            'get_state': get_state,
            'constants': constants
        }

    def mekf_derivation():
        """
        multiplicative kalman filter with quaternions
        :return: dict of equations
        """

        # x, state (7)
        # -----------
        # q, quaternion (4)
        # b, gyro bias (3)
        x = ca.SX.sym('x', 7)
        q = x[:4]
        b_gyro = x[4:7]
        get_state = ca.Function('get_state', [x], [q, b_gyro], ['x'], ['q', 'b_gyro'])

        # state derivative
        xdot = ca.vertcat(quat.kinematics(q, omega_m - b_gyro + w_gyro), w_gyro_rw)
        f_xdot = ca.Function('xdot', [t, x, omega_m, w_gyro, w_gyro_rw],
                             [xdot], ['t', 'x', 'omega_m', 'w_gyro', 'w_gyro_rw'], ['xdot'])

        # state prop w/o noise
        x1 = util.rk4(lambda t, x: f_xdot(t, x, omega_m, ca.DM.zeros(3), ca.DM.zeros(3)), t, x, dt)

        # normalize quaternion
        n_q1 = ca.norm_2(x1[:4])
        x1[0:4] = ca.if_else(ca.fabs(n_q1 - 1) > 1e-6, x1[:4] / n_q1, x1[:4])

        # e, error state (6)
        # ----------------
        # er, so(3) lie algebra rotation error
        # eb, R(3) lie algebra rotation error
        n_e = 6
        eta = ca.SX.sym('eta', n_e, 1)  # (right)
        eta_r = eta[0:3]
        eta_b = eta[3:6]

        # error dynamics
        eta_R = dcm.exp(eta_r)
        f = ca.Function('f', [omega_m, eta, x, w_gyro_rw], [
            ca.vertcat(-ca.mtimes(ca.DM.eye(3) - eta_R, omega_m - b_gyro) - ca.mtimes(eta_R, eta_b),
                       w_gyro_rw)])

        # linearized error dynamics
        F = ca.sparsify(ca.substitute(ca.jacobian(
            f(omega_m, eta, x, w_gyro_rw), eta), eta, ca.SX.zeros(n_e)))

        # covariance propagation
        W = ca.SX.sym('W', ca.Sparsity_lower(n_e))
        f_W_dot_lt = ca.Function(
            'W_dot_lt',
            [x, W, std_gyro, sn_gyro_rw, omega_m, dt],
            [ca.tril(util.sqrt_covariance_predict(W, F, Q))])
        W1 = util.rk4(lambda t, y: f_W_dot_lt(x, y, std_gyro, sn_gyro_rw, omega_m, dt), t, W, dt)

        # prediction
        predict = ca.Function('predict', [t, x, W, omega_m, std_gyro, sn_gyro_rw, dt], [x1, W1],
                              ['t', 'x', 'W', 'omega_m', 'std_gyro', 'sn_gyro_rw', 'dt'], ['x1', 'W1'])

        # mag correction
        y_mag = ca.SX.sym('y_mag', 3, 1)
        correct_mag = ca.Function('correct_mag', [x, W, y_mag], [x, 0.9 * W])

        # accel correction
        y_accel = ca.SX.sym('y_accel', 3, 1)
        correct_accel = ca.Function('correct_accel', [x, W, y_accel], [x, 0.9 * W])

        # initial state
        x0 = ca.DM([1, 0, 0, 0, 0, 0, 0])
        W0 = 1e-3 * ca.DM.eye(n_e)
        constants = ca.Function('constants', [], [x0, W0], [], ['x0', 'W0'])

        return {
            'predict': predict,
            'correct_mag': correct_mag,
            'correct_accel': correct_accel,
            'get_state': get_state,
            'constants': constants
        }

    return {
        'sim': sim_derivation(),
        'mekf': mekf_derivation(),
        'quat': quat_derivation(),
        'mrp': mrp_derivation()
    }