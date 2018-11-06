import multiprocessing as mp

import numpy as np

from pyecca2 import uros
from pyecca2 import replay
from pyecca2.estimators.attitude import algorithms
from pyecca2.estimators.attitude.estimator import AttitudeEstimator
from pyecca2.estimators.attitude.simulator import Simulator

default_params = {
    't0': 0,
    'tf': 1,
    'n_monte_carlo': 1,
    'replay_log_file': None,
    'name': 'default',
    'initialize': True,
    'estimators': [],
    'x0': [0, 0, 0, 0, 0, 0],
    'params': {}
}


def launch_sim(params):
    p = dict(default_params)
    for k, v in params.items():
        if k not in p.keys():
            raise KeyError(k)
        p[k] = v

    print('x0', p['x0'])

    eqs = algorithms.eqs()
    core = uros.Core()
    Simulator(core, eqs, p['x0'])

    for name in p['estimators']:
        AttitudeEstimator(core, name, eqs[name], p['initialize'])

    logger = uros.Logger(core)

    core.init_params()
    for k, v in p['params'].items():
        core.set_param(k, v)

    core.run(until=p['tf'])
    print(p['name'], 'done')
    return logger.get_log_as_array()


def launch_monte_carlo_sim(params):
    p = dict(default_params)

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


def launch_replay(params):
    p = dict(default_params)

    for k, v in params.items():
        if k not in p.keys():
            raise KeyError(k)
        p[k] = v

    eqs = algorithms.eqs()
    core = uros.Core()

    replay.ULogReplay(core, p['replay_log_file'])

    for name in p['estimators']:
        AttitudeEstimator(core, name, eqs[name], p['initialize'])

    logger = uros.Logger(core)

    core.init_params()
    for k, v in p['params'].items():
        core.set_param(k, v)

    core.run(until=p['tf'])
    print(p['name'], 'done')
    return logger.get_log_as_array()