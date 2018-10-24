import simpy
import numpy as np
import multiprocessing as mp
import matplotlib.pyplot as plt

import examples.msgs as msgs
import examples.system as sys


class Simulator:

    def __init__(self, core):
        self.core = core
        self.pub_sim = sys.Publisher(core, 'sim_state', msgs.SimState)
        self.pub_imu = sys.Publisher(core, 'imu', msgs.Imu)
        self.sub_params = sys.Subscriber(core, 'params',  msgs.Params, self.params_callback)
        self.msg_sim_state = msgs.SimState()
        self.msg_imu = msgs.Imu()
        simpy.Process(core, self.run())

    def params_callback(self, msg):
        pass

    def run(self):

        while True:
            t = self.core.now

            # publish imu
            self.msg_imu.data['time'] = t
            self.pub_imu.publish(self.msg_imu)

            # publish sim state
            self.msg_sim_state.data['time'] = t
            self.msg_sim_state.data['q'] = [1, 0, 0, 0]
            self.msg_sim_state.data['omega'] = np.array([
                3 * np.sin(2 * np.pi * t),
                2 * np.sin(2 * np.pi * t + 2),
                1 * np.sin(2 * np.pi * t + 5)]) + 0.2 * np.random.randn(3)
            self.pub_sim.publish(self.msg_sim_state)

            yield simpy.Timeout(self.core, 1.0 / 200)


class Estimator1:

    def __init__(self, core):
        self.core = core
        self.sub_imu = sys.Subscriber(core, 'imu', msgs.Imu, self.imu_callback)
        self.pub_est = sys.Publisher(core, 'est1_status', msgs.EstimatorStatus)
        self.msg_est_status = msgs.EstimatorStatus()
        self.sub_params = sys.Subscriber(core, 'params',  msgs.Params, self.params_callback)

    def params_callback(self, msg):
        pass

    def imu_callback(self, msg):
        # publish estimator
        self.msg_est_status.data['time'] = self.core.now
        #self.msg_est_status.data['x'][:3] = [1, 2, 3]
        self.pub_est.publish(self.msg_est_status)


class Estimator2:

    def __init__(self, core):
        self.core = core
        self.sub_imu = sys.Subscriber(core, 'imu', msgs.Imu, self.imu_callback)
        self.pub_est = sys.Publisher(core, 'est2_status', msgs.EstimatorStatus)
        self.msg_est_status = msgs.EstimatorStatus()
        self.sub_params = sys.Subscriber(core, 'params',  msgs.Params, self.params_callback)

    def params_callback(self, msg):
        pass

    def imu_callback(self, msg):
        # publish estimator
        self.msg_est_status.data['time'] = self.core.now
        #self.msg_est_status.data['x'][:3] = [1, 2, 3]
        self.pub_est.publish(self.msg_est_status)


def do_sim(name):
    core = sys.Core()
    Simulator(core)
    Estimator1(core)
    Estimator2(core)
    logger = sys.Logger(core)

    core.set_params({
        'logger/dt': 1.0/200
    })
    core.run(until=1)
    return logger.get_log_as_array()


with mp.Pool(mp.cpu_count()) as pool:
    data = np.array(pool.map(do_sim, range(1)))


def plot():
    for d in data:
        plt.plot(d['time'], d['est1_status']['x'], 'b', alpha=0.1)
        plt.plot(d['time'], d['est2_status']['x'], 'b', alpha=0.1)

    plt.show()

plot()