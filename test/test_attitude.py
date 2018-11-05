import os
import pickle
import numpy as np
import time

from pyecca2.estimators.attitude import algorithms
from pyecca2.estimators.attitude.launch import launch_monte_carlo_sim
from pyecca2.estimators.attitude.plot import plot


script_dir = os.path.abspath(os.path.dirname(__file__))
results_dir = os.path.join(script_dir, 'results', 'attitude')

alpha = 0.5
est_style={
    'sim': {'color': 'k', 'linestyle': '-.', 'linewidth': 1, 'alpha': alpha},
    'mrp': {'color': 'b', 'linestyle': '--', 'linewidth': 2, 'alpha': alpha},
    'quat': {'color': 'g', 'linestyle': '-.', 'linewidth': 1, 'alpha': alpha},
    'mekf': {'color': 'r', 'linestyle': '-.', 'linewidth': 1, 'alpha': alpha},
    'default': {'color': 'm', 'linestyle': '-.', 'linewidth': 1, 'alpha': alpha}
}

tf = 10
params = {
    'n_monte_carlo': 1,
    'tf': tf,
    'estimators': ['mrp', 'quat'],
    'x0': np.array([0.1, 0.2, 0.3, 0, -0.07, 0.04, 0.07]),
    'params': {
        'sim/dt_sim': 1.0 / 400,
        'sim/dt_imu': 1.0 / 200,
        'sim/dt_mag': 1.0 / 50,
        #'mrp/dt_min_mag': 1.0 / 50,
        #'mrp/dt_min_accel': 1.0 / 200,
        #'quat/dt_min_mag': 1.0 / 50,
        #'quat/dt_min_accel': 1.0 / 200,
        'logger/dt': tf/200,
        'sim/enable_noise': True
    }
}

def test_derivation():
    eqs = algorithms.eqs(results_dir=results_dir)
    print('eqs', eqs.keys())


def test_sim():
    start = time.perf_counter()
    data = launch_monte_carlo_sim(params)
    elapsed = time.perf_counter() - start
    print('\n\nsimulation complete')
    print('-'*30)
    print('cpu time\t\t:', np.round(elapsed, 2))
    print('tf\t\t\t\t:', tf)
    print('n monte carlo\t:', params['n_monte_carlo'])
    print('n estimators\t:', len(params['estimators']))
    print('speed ratio\t\t:', np.round(
        len(params['estimators'])*params['n_monte_carlo']*tf/elapsed, 2))

    data_path = os.path.join(results_dir, 'data.pkl')

    os.makedirs(results_dir, exist_ok=True)
    with open(data_path, 'wb') as f:
        pickle.dump(data, f)
    test_plot()
    return data


def test_generate_code():
    eqs = algorithms.eqs()
    algorithms.generate_code(eqs, os.path.join(results_dir, 'code'))


def test_plot():
    data_path = os.path.join(results_dir, 'data.pkl')
    with open(data_path, 'rb') as f:
        data = pickle.load(f)

    plot(data, ground_truth_name='sim', est_names=params['estimators'],
         est_style=est_style, fig_dir=results_dir, t_start=0.1, show=False)
