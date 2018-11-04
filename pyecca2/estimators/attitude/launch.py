import multiprocessing as mp

import numpy as np

from pyecca2 import uros
from pyecca2.estimators.attitude.derivation import derive_equations
from pyecca2.estimators.attitude.estimator import AttitudeEstimator
from pyecca2.estimators.attitude.simulator import Simulator


def launch_sim(params):
    p = {
        'tf': 1,
        'name': 'default',
        'estimators': [],
        'params': {}
    }
    for k, v in params.items():
        if k not in p.keys():
            raise KeyError(k)
        p[k] = v

    eqs = derive_equations()
    core = uros.Core()
    Simulator(core, eqs)

    for name in p['estimators']:
        AttitudeEstimator(core, name, eqs[name])

    logger = uros.Logger(core)

    core.init_params()
    for k, v in p['params'].items():
        core.set_param(k, v)

    core.run(until=p['tf'])
    print(p['name'], 'done')
    return logger.get_log_as_array()


def launch_monte_carlo_sim(params):
    p = {
        'tf': 1,
        'n_monte_carlo': 1,
        'name': 'default',
        'estimators': [],
        'params': {}
    }
    for k, v in params.items():
        if k not in p.keys():
            raise KeyError(k)
        p[k] = v

    if p['n_monte_carlo'] == 1:
        d = dict(p)
        d.pop('n_monte_carlo')
        data = [launch_sim(d)]
    else:
        new_params = []
        for i in range(p['n_monte_carlo']):
            d = dict(p)
            d.pop('n_monte_carlo')
            d['name'] = i
            new_params.append(d)
        with mp.Pool(mp.cpu_count()) as pool:
            data = np.array(pool.map(launch_sim, new_params))
    return data
