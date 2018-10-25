import simpy
import numpy as np
import multiprocessing as mp
import matplotlib.pyplot as plt

import pyecca2.msgs as msgs
import pyecca2.system as sys
import pyecca2.rotation as rot

import casadi as ca
import time


def rk4(f, t, y, h):
    """Runge Kuta 4th order integrator"""
    k1 = h * f(t, y)
    k2 = h * f(t + h / 2, y + k1 / 2)
    k3 = h * f(t + h / 2, y + k2 / 2)
    k4 = h * f(t + h, y + k3)
    return ca.simplify(y + (k1 + 2 * k2 + 2 * k3 + k4) / 6)


def mrp_derivation():
    # x, state (7)
    #-----------
    # r, mrp (3)
    # s, shadow, mrp shadow state (1)
    # b, gyro bias (3)
    x = ca.SX.sym('x', 7)

    # e, error state (6)
    #----------------
    # er, so(3) lie algebra rotation error
    # eb, R(3) lie algebra rotation error


    r = rot.Mrp(x[0:4])  # last state is shadow state
    b_g = x[4:7]

    omega = ca.SX.sym('omega', 3, 1)
    w_g = ca.SX.sym('w_g', 3, 1)
    w_b = ca.SX.sym('w_b', 3, 1)
    t = ca.SX.sym('t')
    dt = ca.SX.sym('dt')

    # state derivative
    xdot = ca.vertcat(r.derivative(omega - b_g + w_g), w_b)
    f_xdot = ca.Function('xdot', [t, x, omega, w_g, w_b], [xdot], ['t', 'x', 'omega', 'w_g', 'w_b'], ['xdot'])

    # state prop with noise
    x1_noise = rk4(lambda t, x: f_xdot(t, x, omega, w_g, w_b), t, x, dt)
    x1_noise[:4] = rot.Mrp(x1_noise[:4]).shadow_if_required()

    # state prop w/o noise
    x1 = rk4(lambda t, x: f_xdot(t, x, omega, np.zeros(3), np.zeros(3)), t, x, dt)
    x1[:4] = rot.Mrp(x1[:4]).shadow_if_required()

    # quaternion from mrp
    q = rot.Quat.from_mrp(rot.Mrp(x[:4]))

    # constants
    x0 = ca.DM.zeros(7)

    #R = rot.Dcm.from_mrp(r)
    return {
        'x1': ca.Function('f_x1', [t, x, omega, dt], [x1],
                          ['t', 'x', 'omega', 'dt'], ['x1']),
        'x1_noise': ca.Function('f_x1_noise', [t, x, omega, w_g, w_b, dt], [x1_noise],
                          ['t', 'x', 'omega', 'w_g', 'w_b', 'dt'], ['x1']),
        'get_state': ca.Function('get_state', [x], [q, b_g], ['x'], ['q', 'b_g']),
        'constants': ca.Function('constants', [], [x0], [], ['x0'])
    }


mrp_eqs = mrp_derivation()


def quat_derivation():
    # x, state (7)
    #-----------
    # q, quaternion (4)
    # b, gyro bias (3)
    x = ca.SX.sym('x', 7)

    q = rot.Quat(x[:4])
    b_g = x[4:7]

    omega = ca.SX.sym('omega', 3, 1)
    w_g = ca.SX.sym('w_g', 3, 1)
    w_b = ca.SX.sym('w_b', 3, 1)
    t = ca.SX.sym('t')
    dt = ca.SX.sym('dt')

    # state derivative
    xdot = ca.vertcat(q.derivative(omega - b_g + w_g), w_b)
    f_xdot = ca.Function('xdot', [t, x, omega, w_g, w_b], [xdot], ['t', 'x', 'omega', 'w_g', 'w_b'], ['xdot'])

    # state prop w noise
    x1_noise = rk4(lambda t, x: f_xdot(t, x, omega, w_g, w_b), t, x, dt)

    # state prop w/o noise
    x1 = rk4(lambda t, x: f_xdot(t, x, omega, np.zeros(3), np.zeros(3)), t, x, dt)

    # constants
    x0 = ca.DM([1, 0, 0, 0, 0, 0, 0])

    return {
        'x1': ca.Function('f_x1', [t, x, omega, dt], [x1],
                          ['t', 'x', 'omega', 'dt'], ['x1']),
        'x1_noise': ca.Function('f_x1_noise', [t, x, omega, w_g, w_b, dt], [x1],
                          ['t', 'x', 'omega', 'w_g', 'w_b', 'dt'], ['x1']),
        'get_state': ca.Function('get_state', [x], [q, b_g], ['x'], ['q', 'b_g']),
        'constants': ca.Function('constants', [], [x0], [], ['x0'])
    }


quat_eqs = quat_derivation()


class Simulator:

    def __init__(self, core):
        self.core = core
        self.pub_sim = sys.Publisher(core, 'sim_state', msgs.VehicleState)
        self.pub_imu = sys.Publisher(core, 'imu', msgs.Imu)
        self.sub_params = sys.Subscriber(core, 'params', msgs.Params, self.params_callback)
        self.msg_sim_state = msgs.VehicleState()
        self.msg_imu = msgs.Imu()
        self.w_gyro = sys.Param(core, 'sim/w_gyro', 0.1, 'f4')
        self.dt = sys.Param(core, 'sim/dt', 1.0/200, 'f4')
        self.param_list = [self.w_gyro]
        simpy.Process(core, self.run())

    def params_callback(self, msg):
        for p in self.param_list:
            p.update()

    def run(self):
        x = mrp_eqs['constants']()['x0']
        while True:
            t = self.core.now

            # integrate
            omega = np.array([
                10*np.sin(2*np.pi*t + 1),
                20*np.sin(2*np.pi*t + 2),
                30*np.sin(2*np.pi*t + 3)])
            w_g = np.random.randn(3)*1e-3
            w_b = np.random.randn(3)*1e-3
            x = mrp_eqs['x1_noise'](t, x, omega, w_g, w_b, self.dt.get())
            q, b_g = mrp_eqs['get_state'](x)

            # publish sim state
            self.msg_sim_state.data['time'] = t
            self.msg_sim_state.data['q'] = q.T
            self.msg_sim_state.data['b'] = b_g.T
            self.msg_sim_state.data['omega'] = omega
            self.pub_sim.publish(self.msg_sim_state)

            # publish imu
            self.msg_imu.data['time'] = t

            self.msg_imu.data['gyro'] = omega + self.w_gyro.get()*np.random.randn(3)
            self.pub_imu.publish(self.msg_imu)

            yield simpy.Timeout(self.core, self.dt.get())


class AttitudeEstimator:

    def __init__(self, core, name, eqs):
        self.core = core
        self.sub_imu = sys.Subscriber(core, 'imu', msgs.Imu, self.imu_callback)
        self.pub_est = sys.Publisher(core, name + '_status', msgs.EstimatorStatus)
        self.pub_state = sys.Publisher(core, name + '_state', msgs.VehicleState)

        self.msg_est_status = msgs.EstimatorStatus()
        self.msg_state = msgs.VehicleState()

        self.sub_params = sys.Subscriber(core, 'params', msgs.Params, self.params_callback)
        self.param_list = []
        self.x = eqs['constants']()['x0']
        self.n_x = self.x.shape[0]
        self.t_last_imu = 0
        self.eqs = eqs

    def params_callback(self, msg):
        for p in self.param_list:
            p.update()

    def imu_callback(self, msg):

        # compute dt
        t = msg.data['time']
        dt = t - self.t_last_imu
        self.t_last_imu = t
        if dt <= 0:
            dt = 0

        # estimate state
        omega = msg.data['gyro']
        start = time.clock()
        self.x = self.eqs['x1'](t, self.x, omega, dt)
        q, b_g = self.eqs['get_state'](self.x)
        elapsed = time.clock() - start

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
        self.msg_est_status.data['elapsed'] = elapsed
        self.pub_est.publish(self.msg_est_status)


def do_sim(name):
    core = sys.Core()
    Simulator(core)

    AttitudeEstimator(core, 'est1', mrp_eqs)
    AttitudeEstimator(core, 'est2', quat_eqs)

    logger = sys.Logger(core)

    core.run(until=2)
    return logger.get_log_as_array()


with mp.Pool(mp.cpu_count()) as pool:
    data = np.array(pool.map(do_sim, range(1)))


def plot():
    alpha = 0.5

    def compare_field(name_label_style, field, scale=1.0, *args, **kwargs):
        h = {}
        for i, d in enumerate(data):
            for name, label, style in name_label_style:
                h[name] = plt.plot(d['time'], scale*d[name][field],
                                   style, *args, **kwargs)
        plt.legend([ v[0] for k, v in h.items() ], [ v[1] for v in name_label_style])


    plt.figure(1)
    plt.title('q')
    plt.xlabel('time, sec')
    compare_field([
        ('est1_state', 'est1', 'g'),
        ('est2_state', 'est2', 'b'),
        ('sim_state', 'true', 'k'),
    ], 'q', alpha=alpha)
    plt.show()

    plt.figure(2)
    plt.title('quaternion normal error')
    plt.xlabel('time, sec')
    plt.ylabel('normal error')
    h_mrp = None
    h_quat = None
    for d in data:
        n1 = np.linalg.norm(d['est1_state']['q'], axis=1) - 1
        n2 = np.linalg.norm(d['est2_state']['q'], axis=1) - 1
        h_mrp = plt.plot(d['time'], n1, 'b', alpha=alpha)
        h_quat = plt.plot(d['time'], n2, 'g', alpha=alpha)
    plt.legend([h_mrp[0], h_quat[0]], ['mrp', 'quat'])
    plt.show()

    plt.figure(3)
    plt.title('cpu time')
    plt.ylabel('cpu time, usec')
    plt.xlabel('time, sec')
    compare_field([
        ('est1_status', 'est1', 'g'),
        ('est2_status', 'est2', 'b')], 'elapsed',
        scale=1e6, alpha=alpha)
    plt.show()

    plt.figure(4)
    plt.title('angular velocity')
    plt.xlabel('time, sec')
    plt.ylabel('angular velocity, rad/sec')
    for d in data:
        plt.plot(d['time'], d['sim_state']['omega'])
    plt.show()

    plt.figure(5)
    plt.title('bias')
    plt.xlabel('time, sec')
    plt.ylabel('bias, rad/sec')
    compare_field([
        ('est1_state', 'est1', 'g'),
        ('est2_state', 'est2', 'b'),
        ('sim_state', 'true', 'k'),
    ], 'b', alpha=alpha)
    plt.show()

plot()