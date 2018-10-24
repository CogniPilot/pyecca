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
    r = rot.Mrp(ca.SX.sym('r', 4, 1))  # last state is shadow state
    omega = ca.SX.sym('omega', 3, 1)
    rdot = r.derivative(omega)
    t = ca.SX.sym('t')
    h = ca.SX.sym('h')
    f_rdot = ca.Function('rdot', [t, r, omega], [rdot], ['t', 'r', 'omega'], ['rdot'])
    r1 = rot.Mrp(rk4(lambda t, r: f_rdot(t, r, omega), t, r, h)).shadow_if_required()
    q1 = rot.Quat.from_mrp(r1)
    return {
        'r1': ca.Function('f_r1', [t, r, omega, h], [r1, q1], ['t', 'r', 'omega', 'h'], ['r1', 'q1'])
    }


mrp_eqs = mrp_derivation()


def quat_derivation():
    q = rot.Quat(ca.SX.sym('q', 4, 1))  # last state is shadow state
    omega = ca.SX.sym('omega', 3, 1)
    rdot = q.derivative(omega)
    t = ca.SX.sym('t')
    h = ca.SX.sym('h')
    f_qdot = ca.Function('rdot', [t, q, omega], [rdot], ['t', 'q', 'omega'], ['qdot'])
    q1 = rot.Quat(rk4(lambda t, q: f_qdot(t, q, omega), t, q, h))
    return {
        'q1': ca.Function('f_q1', [t, q, omega, h], [q1], ['t', 'q', 'omega', 'h'], ['q1'])
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
        r = np.zeros(4)
        while True:
            t = self.core.now

            # integrate
            omega = np.array([
                10*np.sin(2*np.pi*t + 1),
                20*np.sin(2*np.pi*t + 2),
                30*np.sin(2*np.pi*t + 3)])
            r, q = mrp_eqs['r1'](t, r, omega, self.dt.get())

            # publish sim state
            self.msg_sim_state.data['time'] = t
            self.msg_sim_state.data['q'] = q.T
            self.msg_sim_state.data['r'] = r.T
            self.msg_sim_state.data['omega'] = omega
            self.pub_sim.publish(self.msg_sim_state)

            # publish imu
            self.msg_imu.data['time'] = t

            self.msg_imu.data['gyro'] = omega + self.w_gyro.get()*np.random.randn(3)
            self.pub_imu.publish(self.msg_imu)

            yield simpy.Timeout(self.core, self.dt.get())


class Estimator1:

    def __init__(self, core):
        self.core = core
        self.sub_imu = sys.Subscriber(core, 'imu', msgs.Imu, self.imu_callback)
        self.pub_est = sys.Publisher(core, 'est1_status', msgs.EstimatorStatus)
        self.pub_state = sys.Publisher(core, 'est1_state', msgs.VehicleState)

        self.msg_est_status = msgs.EstimatorStatus()
        self.msg_state = msgs.VehicleState()

        self.sub_params = sys.Subscriber(core, 'params', msgs.Params, self.params_callback)
        self.param_list = []
        self.r = np.zeros(4)
        self.t_last_imu = 0

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
        self.r, q = mrp_eqs['r1'](t, self.r, omega, dt)
        elapsed = time.clock() - start

        # publish vehicle state
        self.msg_state.data['time'] = t
        self.msg_state.data['q'] = q.T
        self.msg_state.data['omega'] = omega.T
        self.pub_state.publish(self.msg_state)

        # publish estimator status
        self.msg_est_status.data['time'] = t
        self.msg_est_status.data['x'][:4] = self.r.T
        self.msg_est_status.data['elapsed'] = elapsed
        self.pub_est.publish(self.msg_est_status)


class Estimator2:

    def __init__(self, core):
        self.core = core
        self.sub_imu = sys.Subscriber(core, 'imu', msgs.Imu, self.imu_callback)
        self.pub_est = sys.Publisher(core, 'est2_status', msgs.EstimatorStatus)
        self.pub_state = sys.Publisher(core, 'est2_state', msgs.VehicleState)

        self.msg_est_status = msgs.EstimatorStatus()
        self.msg_state = msgs.VehicleState()

        self.sub_params = sys.Subscriber(core, 'params', msgs.Params, self.params_callback)
        self.param_list = []
        self.q = np.array([1, 0, 0, 0])
        self.t_last_imu = 0

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
        self.q = quat_eqs['q1'](t, self.q, omega, dt)
        elapsed = time.clock() - start

        # publish vehicle state
        self.msg_state.data['time'] = t
        self.msg_state.data['q'] = self.q.T
        self.msg_state.data['omega'] = omega.T
        self.pub_state.publish(self.msg_state)

        # publish estimator status
        self.msg_est_status.data['time'] = t
        self.msg_est_status.data['x'][:4] = self.q.T
        self.msg_est_status.data['elapsed'] = elapsed
        self.pub_est.publish(self.msg_est_status)


def do_sim(name):
    core = sys.Core()
    Simulator(core)

    Estimator1(core)
    Estimator2(core)

    logger = sys.Logger(core)

    core.run(until=2)
    return logger.get_log_as_array()


with mp.Pool(mp.cpu_count()) as pool:
    data = np.array(pool.map(do_sim, range(1)))


def plot():
    plt.figure(1)
    plt.title('q')
    for d in data:
        plt.plot(d['time'], d['sim_state']['q'])
    plt.show()

    plt.figure(2)
    plt.title('r')
    for d in data:
        plt.plot(d['time'], d['sim_state']['r'])
    plt.show()

    plt.figure(2)
    plt.title('quaternion normal error')
    plt.xlabel('time, sec')
    plt.ylabel('normal error')

    for d in data:
        n1 = np.linalg.norm(d['est1_state']['q'], axis=1) - 1
        n2 = np.linalg.norm(d['est2_state']['q'], axis=1) - 1
        plt.plot(d['time'], n1, 'b', label='mrp')
        plt.plot(d['time'], n2, 'g', label='quat')
    plt.legend()
    plt.show()

    plt.figure(3)
    plt.title('cpu time')
    plt.ylabel('cpu time, usec')
    plt.xlabel('time, sec')
    for d in data:
        plt.plot(d['time'], 1e6*d['est1_status']['elapsed'], alpha=0.5, label='mrp')
        plt.plot(d['time'], 1e6*d['est2_status']['elapsed'], alpha=0.5, label='quat')
    plt.legend()

    plt.show()


    plt.figure(4)
    plt.title('angular velocity')
    plt.xlabel('time, sec')
    plt.ylabel('angular velocity, rad/sec')
    for d in data:
        plt.plot(d['time'], d['sim_state']['omega'])
    plt.show()

plot()