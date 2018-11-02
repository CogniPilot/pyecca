import multiprocessing as mp

import numpy as np

from pyecca2 import uros
from pyecca2.estimators.attitude.derivation import derivation
from pyecca2.estimators.attitude.estimator import AttitudeEstimator
from pyecca2.estimators.attitude.simulator import Simulator


def launch_sim(params):
    tf = params['tf']
    sim_name = params['name']
    eqs = derivation()
    core = uros.Core()
    Simulator(core, eqs)

    estimators = [
        ('est1', eqs['mekf']),
        ('est2', eqs['quat']),
        ('est3', eqs['mrp'])
    ]

    for name, eqs in estimators:
        AttitudeEstimator(core, name, eqs)

    logger = uros.Logger(core)

    core.init_params()
    core.set_param('logger/dt', 1.0/200)
    core.set_param('sim/enable_noise', True)

    core.run(until=tf)
    print(sim_name, 'done')
    return logger.get_log_as_array()


def launch_monte_carlo_sim(params):
    tf = params['tf']
    n = params['n']
    if params['n'] == 1:
        data = [launch_sim({'name': 0, 'tf': tf})]
    else:
        new_params = []
        for i in range(n):
            new_params.append({'name': i, 'tf': tf})
        with mp.Pool(mp.cpu_count()) as pool:
            data = np.array(pool.map(launch_sim, new_params))
    return data
