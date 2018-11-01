import multiprocessing as mp

import numpy as np

from pyecca2 import uros
from pyecca2.estimators.attitude.derivation import derivation
from pyecca2.estimators.attitude.estimator import AttitudeEstimator
from pyecca2.estimators.attitude.simulator import Simulator


def launch_sim(sim_name):
    tf = 10
    eqs = derivation()
    core = uros.Core()
    Simulator(core, eqs)

    for name, eqs in [('est1', eqs['mekf']), ('est2', eqs['quat']), ('est3', eqs['mrp'])]:
        AttitudeEstimator(core, name, eqs)

    logger = uros.Logger(core)

    core.run(until=tf)
    print(sim_name, 'done')
    return logger.get_log_as_array()


def launch_monte_carlo_sim(n=1):
    if n == 1:
        data = [launch_sim(0)]
    else:
        with mp.Pool(mp.cpu_count()) as pool:
            data = np.array(pool.map(launch_sim, range(n)))
    return data
