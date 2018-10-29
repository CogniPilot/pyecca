#%%
import sys
import os.path
import time
import multiprocessing as mp

import simpy
import numpy as np
import matplotlib.pyplot as plt
import casadi as ca

import pyecca2.msgs as msgs
import pyecca2.system as sys
import pyecca2.rotation as rot
import pyecca2.util as util


def derivation():

    # misc variables
    omega = ca.SX.sym('omega', 3, 1) # angular velocity in body frame
    t = ca.SX.sym('t')  # time
    dt = ca.SX.sym('dt')  # integration time step

    # magnetic field
    decl = ca.SX.sym('decl')
    incl = ca.SX.sym('incl')  # only useful for sim, neglected in correction

    # noise
    w_g = ca.SX.sym('w_g', 3, 1)
    w_b = ca.SX.sym('w_b', 3, 1)
    std_gyro = ca.SX.sym('std_gyro')
    std_mag = ca.SX.sym('std_mag')
    sn_gyro_rw = ca.SX.sym('sn_gyro_rw')
    std_gyro_rw = sn_gyro_rw / ca.sqrt(dt)
    Q = ca.diag(ca.vertcat(std_gyro, std_gyro, std_gyro, std_gyro_rw, std_gyro_rw, std_gyro_rw) ** 2)

    e1 = ca.SX([1, 0, 0])
    e2 = ca.SX([0, 1, 0])
    e3 = ca.SX([0, 0, 1])

    def sim_derivation():

        # x, state (7)
        #-----------
        # r, mrp (3)
        # s, shadow, mrp shadow state (1)
        # b, gyro bias (3)
        x = ca.SX.sym('x', 7)
        r = rot.Mrp(x[0:4])  # last state is shadow state
        b_g = x[4:7]

        # initial state
        x0 = ca.DM.zeros(7)

        # state derivative
        xdot = ca.vertcat(r.derivative(omega - b_g + w_g), w_b)
        f_xdot = ca.Function('xdot', [t, x, omega, w_g, w_b], [xdot], ['t', 'x', 'omega', 'w_g', 'w_b'], ['xdot'])

        # state prop with noise
        x1_sim = util.rk4(lambda t, x: f_xdot(t, x, omega, w_g, w_b), t, x, dt)
        x1_sim[:4] = rot.Mrp(x1_sim[:4]).shadow_if_required()

        # state prop w/o noise
        x1 = util.rk4(lambda t, x: f_xdot(t, x, omega, np.zeros(3), np.zeros(3)), t, x, dt)
        x1[:4] = rot.Mrp(x1[:4]).shadow_if_required()

        # quaternion from mrp
        q = rot.Quat.from_mrp(rot.Mrp(x[:4]))
        C_nb = rot.Dcm.from_mrp(r)

        # measure_mag
        B_n = ca.mtimes(rot.SO3.exp(decl * e3) * rot.SO3.exp(-incl * e2), ca.SX([1, 0, 0]))
        measure_mag = ca.Function(
            'measure_mag', [x, decl, incl], [ca.mtimes(C_nb.T, B_n)], ['x', 'decl', 'incl'], ['y'])

        # measure accel
        measure_accel = ca.Function(
            'measure_accel', [x], [ca.mtimes(C_nb.T, ca.SX([0, 0, -9.8]))], ['x'], ['y'])

        return {
            'simulate': ca.Function('simulate', [t, x, omega, w_g, w_b, dt], [x1_sim],
                              ['t', 'x', 'omega', 'w_g', 'w_b', 'dt'], ['x1']),
            'measure_mag': measure_mag,
            'measure_accel': measure_accel,
            'get_state': ca.Function('get_state', [x], [q, b_g], ['x'], ['q', 'b_g']),
            'constants': ca.Function('constants', [], [x0], [], ['x0'])
        }

    def mrp_derivation():
        # mrp, RIEKF
        # right invariant kalman filter with modified rodrigues parameters


        # x, state (7)
        #-----------
        # r, mrp (3)
        # s, shadow, mrp shadow state (1)
        # b, gyro bias (3)
        x = ca.SX.sym('x', 7)
        r = rot.Mrp(x[0:4])  # last state is shadow state
        b_g = x[4:7]

        # state derivative
        xdot = ca.vertcat(r.derivative(omega - b_g + w_g), w_b)
        f_xdot = ca.Function('xdot', [t, x, omega, w_g, w_b], [xdot], ['t', 'x', 'omega', 'w_g', 'w_b'], ['xdot'])

        # state prop w/o noise
        x1 = util.rk4(lambda t, x: f_xdot(t, x, omega, np.zeros(3), np.zeros(3)), t, x, dt)
        x1[:4] = rot.Mrp(x1[:4]).shadow_if_required()

        # quaternion from mrp
        q = rot.Quat.from_mrp(rot.Mrp(x[:4]))

        # e, error state (6)
        #----------------
        # er, so(3) lie algebra rotation error
        # eb, R(3) lie algebra rotation error
        n_e = 6
        eta = ca.SX.sym('eta', n_e, 1)  # (right)

        # error dynamics
        f = ca.Function('f', [omega, eta, x, w_b], [
            ca.vertcat(-ca.mtimes(rot.Dcm.from_mrp(r), eta[3:6]), w_b)])

        # linearized error dynamics
        F = ca.sparsify(ca.substitute(ca.jacobian(f(omega, eta, x, w_b), eta), eta, ca.SX.zeros(n_e)))

        # covariance propagation
        W = ca.SX.sym('W', ca.Sparsity_lower(n_e))
        f_W_dot_lt = ca.Function(
            'W_dot_lt',
            [x, W, std_gyro, sn_gyro_rw, omega, dt],
            [ca.tril(util.sqrt_covariance_predict(W, F, Q))])
        W1 = util.rk4(lambda t, y: f_W_dot_lt(x, y, std_gyro, sn_gyro_rw, omega, dt), t, W, dt)

        # initial state
        x0 = ca.DM.zeros(7)
        W0 = 1e-3*np.eye(n_e)

        return {
            'predict': ca.Function('predict', [t, x, W, omega, std_gyro, sn_gyro_rw, dt], [x1, W1],
                              ['t', 'x', 'W', 'omega', 'std_gyro', 'sn_gyro_rw', 'dt'], ['x1', 'W1']),
            'get_state': ca.Function('get_state', [x], [q, b_g], ['x'], ['q', 'b_g']),
            'constants': ca.Function('constants', [], [x0, W0], [], ['x0', 'W0'])
        }

    def quat_derivation():
        # right invariant kalman filter with quaternions

        # x, state (7)
        #-----------
        # q, quaternion (4)
        # b, gyro bias (3)
        x = ca.SX.sym('x', 7)
        q = rot.Quat(x[:4])
        b_g = x[4:7]

        # state derivative
        xdot = ca.vertcat(q.derivative(omega - b_g + w_g), w_b)
        f_xdot = ca.Function('xdot', [t, x, omega, w_g, w_b], [xdot], ['t', 'x', 'omega', 'w_g', 'w_b'], ['xdot'])

        # state prop w/o noise
        x1 = util.rk4(lambda t, x: f_xdot(t, x, omega, np.zeros(3), np.zeros(3)), t, x, dt)
        n_q1 = ca.norm_2(x1[:4])

        # normalize quaternion
        x1[0:4] = ca.if_else(np.fabs(n_q1 - 1) > 1e-6, x1[:4]/n_q1, x1[:4])

        # e, error state (6)
        #----------------
        # er, so(3) lie algebra rotation error
        # eb, R(3) lie algebra rotation error
        n_e = 6
        eta = ca.SX.sym('eta', n_e, 1)  # (right)

        # error dynamics
        f = ca.Function('f', [omega, eta, x, w_b], [
            ca.vertcat(-ca.mtimes(rot.Dcm.from_quat(q), eta[3:6]), w_b)])

        # linearized error dynamics
        F = ca.sparsify(ca.substitute(ca.jacobian(f(omega, eta, x, w_b), eta), eta, ca.SX.zeros(n_e)))

        # covariance propagation
        W = ca.SX.sym('W', ca.Sparsity_lower(n_e))
        f_W_dot_lt = ca.Function(
            'W_dot_lt',
            [x, W, std_gyro, sn_gyro_rw, omega, dt],
            [ca.tril(util.sqrt_covariance_predict(W, F, Q))])
        W1 = util.rk4(lambda t, y: f_W_dot_lt(x, y, std_gyro, sn_gyro_rw, omega, dt), t, W, dt)

        # initial state
        x0 = ca.DM([1, 0, 0, 0, 0, 0, 0])
        W0 = 1e-3*np.eye(n_e)

        return {
            'predict': ca.Function('predict', [t, x, W, omega, std_gyro, sn_gyro_rw, dt], [x1, W1],
                              ['t', 'x', 'W', 'omega', 'std_gyro', 'sn_gyro_rw', 'dt'], ['x1', 'W1']),
            'get_state': ca.Function('get_state', [x], [q, b_g], ['x'], ['q', 'b_g']),
            'constants': ca.Function('constants', [], [x0, W0], [], ['x0', 'W0'])
        }

    def mekf_derivation():
        # multiplicative kalman filter with quaternions

        # x, state (7)
        # -----------
        # q, quaternion (4)
        # b, gyro bias (3)
        x = ca.SX.sym('x', 7)
        q = rot.Quat(x[:4])
        b_g = x[4:7]

        # state derivative
        xdot = ca.vertcat(q.derivative(omega - b_g + w_g), w_b)
        f_xdot = ca.Function('xdot', [t, x, omega, w_g, w_b], [xdot], ['t', 'x', 'omega', 'w_g', 'w_b'], ['xdot'])

        # state prop w/o noise
        x1 = util.rk4(lambda t, x: f_xdot(t, x, omega, np.zeros(3), np.zeros(3)), t, x, dt)

        # normalize quaternion
        n_q1 = ca.norm_2(x1[:4])
        x1[0:4] = ca.if_else(np.fabs(n_q1 - 1) > 1e-6, x1[:4] / n_q1, x1[:4])

        # e, error state (6)
        # ----------------
        # er, so(3) lie algebra rotation error
        # eb, R(3) lie algebra rotation error
        n_e = 6
        eta = ca.SX.sym('eta', n_e, 1)  # (right)
        eta_r = eta[0:3]
        eta_b = eta[3:6]

        # error dynamics
        eta_R = rot.SO3.exp(eta_r)
        f = ca.Function('f', [omega, eta, x, w_b], [
            ca.vertcat(-ca.mtimes(np.eye(3) - eta_R, omega - b_g) - ca.mtimes(eta_R, eta_b), w_b)])

        # linearized error dynamics
        F = ca.sparsify(ca.substitute(ca.jacobian(f(omega, eta, x, w_b), eta), eta, ca.SX.zeros(n_e)))

        # covariance propagation
        W = ca.SX.sym('W', ca.Sparsity_lower(n_e))
        f_W_dot_lt = ca.Function(
            'W_dot_lt',
            [x, W, std_gyro, sn_gyro_rw, omega, dt],
            [ca.tril(util.sqrt_covariance_predict(W, F, Q))])
        W1 = util.rk4(lambda t, y: f_W_dot_lt(x, y, std_gyro, sn_gyro_rw, omega, dt), t, W, dt)

        # initial state
        x0 = ca.DM([1, 0, 0, 0, 0, 0, 0])
        W0 = 1e-3 * np.eye(n_e)

        return {
            'predict': ca.Function('predict', [t, x, W, omega, std_gyro, sn_gyro_rw, dt], [x1, W1],
                                   ['t', 'x', 'W', 'omega', 'std_gyro', 'sn_gyro_rw', 'dt'], ['x1', 'W1']),
            'get_state': ca.Function('get_state', [x], [q, b_g], ['x'], ['q', 'b_g']),
            'constants': ca.Function('constants', [], [x0, W0], [], ['x0', 'W0'])
        }

    return {
        'sim': sim_derivation(),
        'mekf': mekf_derivation(),
        'quat': quat_derivation(),
        'mrp': mrp_derivation()
    }


class Simulator:

    def __init__(self, core, eqs):
        self.core = core

        # publications
        self.pub_sim = sys.Publisher(core, 'sim_state', msgs.VehicleState)
        self.pub_imu = sys.Publisher(core, 'imu', msgs.Imu)
        self.pub_mag = sys.Publisher(core, 'mag', msgs.Mag)

        # subscriptions
        self.sub_params = sys.Subscriber(core, 'params', msgs.Params, self.params_callback)

        # parameters
        self.w_gyro = sys.Param(core, 'sim/w_gyro', 1e-6, 'f4')
        self.w_gyro_rw = sys.Param(core, 'sim/w_gyro_rw', 1e-6, 'f4')
        self.dt = sys.Param(core, 'sim/dt', 1.0/200, 'f4')
        self.decl = sys.Param(core, 'sim/decl', 0, 'f4')
        self.incl = sys.Param(core, 'sim/incl', 0, 'f4')

        # msgs
        self.msg_sim_state = msgs.VehicleState()
        self.msg_imu = msgs.Imu()
        self.msg_mag = msgs.Mag()

        # misc
        self.enable_noise = True
        self.param_list = [self.w_gyro]
        self.eqs = eqs
        np.random.seed()
        simpy.Process(core, self.run())

    def params_callback(self, msg):
        for p in self.param_list:
            p.update()

    def randn(self, *args, **kwargs):
        return np.random.randn(*args)*self.enable_noise

    def run(self):
        x = self.eqs['sim']['constants']()['x0']
        i = 0
        while True:
            t = self.core.now
            i += 1

            # integrate
            omega = np.array([
                10*(1 + np.sin(2*np.pi*1*t + 1)),
                11*(1 + np.sin(2*np.pi*2*t + 2)),
                12*(1 + np.sin(2*np.pi*3*t + 3))])
            w_g = self.randn(3)*self.w_gyro.get()
            w_b = self.randn(3)*self.w_gyro_rw.get()
            x = self.eqs['sim']['simulate'](t, x, omega, w_g, w_b, self.dt.get())
            q, b_g = self.eqs['sim']['get_state'](x)

            # publish sim state
            self.msg_sim_state.data['time'] = t
            self.msg_sim_state.data['q'] = q.T
            self.msg_sim_state.data['b'] = b_g.T
            self.msg_sim_state.data['omega'] = omega
            self.pub_sim.publish(self.msg_sim_state)

            # publish imu
            self.msg_imu.data['time'] = t
            self.msg_imu.data['gyro'] = omega + w_g
            self.msg_imu.data['accel'] = self.eqs['sim']['measure_accel'](x).T
            self.pub_imu.publish(self.msg_imu)

            # publish mag
            self.msg_mag.data['time'] = t
            self.msg_mag.data['mag'] = self.eqs['sim']['measure_mag'](x, self.decl.get(), self.incl.get()).T
            self.pub_mag.publish(self.msg_mag)

            yield simpy.Timeout(self.core, self.dt.get())


class AttitudeEstimator:

    def __init__(self, core, name, eqs):
        self.core = core

        # subscriptions
        self.sub_imu = sys.Subscriber(core, 'imu', msgs.Imu, self.imu_callback)
        self.sub_mag = sys.Subscriber(core, 'mag', msgs.Mag, self.mag_callback)

        # publications
        self.pub_est = sys.Publisher(core, name + '_status', msgs.EstimatorStatus)
        self.pub_state = sys.Publisher(core, name + '_state', msgs.VehicleState)

        self.msg_est_status = msgs.EstimatorStatus()
        self.msg_state = msgs.VehicleState()

        self.sub_params = sys.Subscriber(core, 'params', msgs.Params, self.params_callback)
        self.param_list = []
        self.x = eqs['constants']()['x0']
        self.W = eqs['constants']()['W0']
        self.n_x = self.x.shape[0]
        self.n_e = self.W.shape[0]
        self.t_last_imu = 0
        self.eqs = eqs

    def params_callback(self, msg):
        for p in self.param_list:
            p.update()

    def mag_callback(self, msg):
        pass

    def imu_callback(self, msg):

        # compute dt
        t = msg.data['time']
        dt = t - self.t_last_imu
        self.t_last_imu = t
        if dt <= 0:
            dt = 1.0/200

        # estimate state
        omega = msg.data['gyro']
        start = time.thread_time()
        std_gyro = 1e-2
        sn_gyro_rw = 1e-2
        self.x, self.W = self.eqs['predict'](t, self.x, self.W, omega, std_gyro, sn_gyro_rw, dt)
        q, b_g = self.eqs['get_state'](self.x)
        end = time.thread_time()
        elapsed = end - start

        # correct

        # publish vehicle state
        self.msg_state.data['time'] = t
        self.msg_state.data['q'] = q.T
        self.msg_state.data['b'] = b_g.T
        self.msg_state.data['omega'] = omega.T
        self.pub_state.publish(self.msg_state)

        # publish estimator status
        self.msg_est_status.data['time'] = t
        self.msg_est_status.data['n_x'] = self.n_x
        self.msg_est_status.data['x'][:self.n_x] = self.x.T
        W_vect = np.reshape(np.array(self.W)[np.diag_indices(self.n_e)], -1)
        self.msg_est_status.data['W'][:len(W_vect)] = W_vect
        self.msg_est_status.data['elapsed'] = elapsed
        self.pub_est.publish(self.msg_est_status)


def do_sim(sim_name, eqs, tf):
    core = sys.Core()
    Simulator(core, eqs)

    for name, eqs in [('est1', eqs['mekf']), ('est2', eqs['quat']), ('est3', eqs['mrp'])]:
        AttitudeEstimator(core, name, eqs)

    logger = sys.Logger(core)

    core.run(until=tf)
    print(sim_name, 'done')
    return logger.get_log_as_array()


def mc_sim(tf, n=1):
    eqs = derivation()
    n_sim = 1

    if n_sim == 1:
        data = [do_sim(0, eqs, tf)]
    else:
        with mp.Pool(mp.cpu_count()) as pool:
            data = np.array(pool.map(do_sim, range(100), eqs, tf))
    return data


def plot(data):

    import os
    if not os.path.exists('fig'):
        os.mkdir('fig')

    est_style = {
        'true': {'color': 'k', 'linewidth': 2, 'linestyle': '-', 'alpha': 0.5},
        'mrp': {'color': 'b', 'linewidth': 2, 'linestyle': '-.', 'alpha': 0.5},
        'quat': {'color': 'g', 'linewidth': 2, 'linestyle': ':', 'alpha': 0.5},
        'mekf': {'color': 'r', 'linewidth': 2, 'linestyle': '--', 'alpha': 0.5},
    }

    label_map = {
        'sim_state': 'true',
        'est1_state': 'mekf',
        'est2_state': 'quat',
        'est3_state': 'mrp',
        'est1_status': 'mekf',
        'est2_status': 'quat',
        'est3_status': 'mrp',
    }

    def compare_topics(topics, get_data, *args, **kwargs):
        h = {}
        for d in data:
            for topic in topics:
                label = label_map[topic]
                h[topic] = plt.plot(d['time'], get_data(d, topic),
                                   *args, **est_style[label], **kwargs)
        plt.legend([ v[0] for k, v in h.items() ], [ label_map[topic] for topic in topics])
        plt.grid()

    plt.figure()
    plt.title('quaternion normal error')
    plt.xlabel('time, sec')
    plt.ylabel('normal error')
    plt.grid()
    compare_topics(['est1_state', 'est2_state', 'est3_state'],
                   lambda data, topic: np.linalg.norm(data[topic]['q'], axis=1) - 1)
    plt.savefig('fig/quat_normal.png')

    plt.figure()
    plt.title('cpu time')
    plt.ylabel('cpu time, usec')
    plt.xlabel('time, sec')
    plt.grid()
    compare_topics(['est1_status', 'est2_status', 'est3_status'],
                   lambda data, topic: 1e6*data[topic]['elapsed'])
    plt.savefig('fig/cpu_time.png')

    plt.figure()
    plt.title('rotation error')
    plt.xlabel('time, sec')
    plt.ylabel('error, deg')
    plt.grid()
    def compare_rot_error(q1, q2):
        r = []
        for q1i, q2i in zip(q1, q2):
            R1 = rot.Dcm.from_quat(rot.Quat(q1i))
            R2 = rot.Dcm.from_quat(rot.Quat(q2i))
            dR = rot.SO3(R1.T*R2)
            ri = np.linalg.norm(ca.DM(rot.SO3.log(dR)))
            r.append(ri)
        r = np.rad2deg(np.array(r))
        return r
    compare_topics(['est1_state', 'est2_state', 'est3_state'],
                   lambda data, topic: compare_rot_error(data[topic]['q'], data['sim_state']['q']))
    plt.savefig('fig/rotation_error.png')

    plt.figure()
    plt.title('angular velocity')
    plt.xlabel('time, sec')
    plt.ylabel('angular velocity, deg/sec')
    plt.grid()
    for d in data:
        plt.plot(d['time'], np.rad2deg(d['sim_state']['omega']))
    plt.savefig('fig/angular_velocity.png')

    plt.figure()
    plt.title('bias')
    plt.xlabel('time, sec')
    plt.ylabel('bias, deg/sec')
    plt.grid()
    compare_topics(['sim_state', 'est1_state', 'est2_state', 'est3_state'],
                   lambda data, topic: np.rad2deg(data[topic]['b']))
    plt.savefig('fig/bias.png')

    plt.figure()
    plt.title('fig/state uncertainty')
    plt.xlabel('time, sec')
    plt.ylabel('std. deviation')
    plt.grid()
    compare_topics(['est1_status', 'est2_status', 'est3_status'],
                   lambda data, topic: data[topic]['W'][:, :3])
    plt.savefig('fig/state_uncertainty.png')

    plt.figure()
    plt.title('mag')
    plt.xlabel('time, sec')
    plt.ylabel('magnetometer, normalized')
    plt.grid()
    for d in data:
        plt.plot(d['time'], d['mag']['mag'])
    plt.savefig('fig/mag.png')

    plt.figure()
    plt.title('accel')
    plt.xlabel('time, sec')
    plt.ylabel('accelerometer, m/s^2')
    plt.grid()
    for d in data:
        plt.plot(d['time'], d['imu']['accel'])
    plt.savefig('fig/accel.png')

def test_sim():
    data = mc_sim(tf=10, n=1)
    plot(data)
