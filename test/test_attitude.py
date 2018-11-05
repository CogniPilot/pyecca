import os
import pickle
import numpy as np

from pyecca2.estimators.attitude import derivation
from pyecca2.estimators.attitude.launch import launch_monte_carlo_sim
from pyecca2.estimators.attitude.plot import plot


script_dir = os.path.abspath(os.path.dirname(__file__))
results_dir = os.path.join(script_dir, 'results', 'attitude')

est_style={
    'sim': {'color': 'k', 'linestyle': '-.', 'linewidth': 2, 'alpha': 1},
    'mrp': {'color': 'b', 'linestyle': '--', 'linewidth': 2, 'alpha': 1},
    'quat': {'color': 'g', 'linestyle': '-.', 'linewidth': 2, 'alpha': 1},
    'mekf': {'color': 'r', 'linestyle': '-.', 'linewidth': 2, 'alpha': 1},
    'default': {'color': 'm', 'linestyle': '-.', 'linewidth': 2, 'alpha': 1}
}

tf = 10
params = {
    'n_monte_carlo': 1,
    'tf': tf,
    'estimators': ['mrp', 'quat'],
    'x0': np.array([0.1, 0.2, 0.3, 0, 0.01, 0.02, 0.03]),
    'f_omega': lambda t: 10 * np.array([np.cos(t), np.sin(t), np.cos(t)]),
    'params': {
        'sim/dt_sim': 1.0 / 400,
        'sim/dt_imu': 1.0 / 200,
        'sim/dt_mag': 1.0 / 50,
        'logger/dt': 1.0 / 100,
        'sim/enable_noise': False
    }
}


def test_derivation():
    eqs = derivation.derive_equations()
    print('eqs', eqs.keys())


def test_sim():
    data = launch_monte_carlo_sim(params)
    data_path = os.path.join(results_dir, 'data.pkl')

    os.makedirs(results_dir, exist_ok=True)
    with open(data_path, 'wb') as f:
        pickle.dump(data, f)
    return data


def test_generate_code():
    eqs = derivation.derive_equations()
    derivation.generate_code(eqs, os.path.join(results_dir, 'code'))


def test_plot():
    data_path = os.path.join(results_dir, 'data.pkl')
    with open(data_path, 'rb') as f:
        data = pickle.load(f)

    plot(data, ground_truth_name='sim', est_names=params['estimators'],
         est_style=est_style, fig_dir=results_dir, t_start=1, show=False)
