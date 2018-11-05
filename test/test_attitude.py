import os
import pickle
import numpy as np

from pyecca2.estimators.attitude import derivation
from pyecca2.estimators.attitude.launch import launch_monte_carlo_sim
from pyecca2.estimators.attitude.plot import plot


def test_derivation():
    eqs = derivation.derive_equations()
    print('eqs', eqs.keys())


def test_sim():
    script_dir = os.path.abspath(os.path.dirname(__file__))
    results_dir = os.path.join(script_dir, 'results', 'attitude')
    tf = 10
    params = {
        'n_monte_carlo': 1,
        'tf': tf,
        'estimators': ['mrp'],
        'x0': np.array([0.1, 0.2, 0.3, 0, 0.01, 0.02, 0.03]),
        'f_omega': lambda t: 1*np.array([np.cos(t), np.sin(t), np.cos(t)]),
        'params': {
            'sim/dt_sim': 1.0/400,
            'sim/dt_imu': 1.0/200,
            'sim/dt_mag': 1.0/50,
            'logger/dt': 1.0/200,
            'sim/enable_noise': True
        }
    }
    data = launch_monte_carlo_sim(params)
    data_path = os.path.join(results_dir, 'data.pkl')

    os.makedirs(results_dir, exist_ok=True)
    with open(data_path, 'wb') as f:
        pickle.dump(data, f)

    with open(data_path, 'rb') as f:
        data = pickle.load(f)

    plot(data, ground_truth_name='sim', est_names=params['estimators'], est_style={
        'sim': {'color': 'k', 'linestyle': '-', 'alpha': 0.5},
        'mrp': {'color': 'b', 'linestyle': '--', 'alpha': 0.5},
        'quat': {'color': 'g', 'linestyle': '-.', 'alpha': 0.5},
        'mekf': {'color': 'r', 'linestyle': '-.', 'alpha': 0.5},
        'default': {'color': 'm', 'linestyle': '-.', 'alpha': 0.5}
        }, fig_dir=results_dir, t_start=0, show=False)


    eqs = derivation.derive_equations()
    derivation.generate_code(eqs, os.path.join(results_dir, 'code'))

    return data


if __name__ == "__main__":
    data = test_sim()