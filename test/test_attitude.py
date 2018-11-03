import os
import pickle

from pyecca2.estimators.attitude.derivation import derivation
from pyecca2.estimators.attitude.launch import launch_monte_carlo_sim
from pyecca2.estimators.attitude.plot import plot


def test_derivation():
    eqs = derivation()
    print('eqs', eqs.keys())


def test_sim():
    script_dir = os.path.abspath(os.path.dirname(__file__))
    results_dir = os.path.join(script_dir, 'results', 'attitude')
    tf = 10
    params = {
        'n_monte_carlo': 1,
        'tf': tf,
        'estimators': ['mrp'],
        'params': {
            'sim/dt_sim': 1.0/400,
            'sim/dt_mag': 1.0/50,
            'logger/dt': tf/100,
            'sim/enable_noise': False
        }
    }
    data = launch_monte_carlo_sim(params)
    data_path = os.path.join(results_dir, 'data.pkl')

    with open(data_path, 'wb') as f:
        pickle.dump(data, f)

    with open(data_path, 'rb') as f:
        data = pickle.load(f)

    plot(data, ground_truth_name='sim', est_names=params['estimators'], est_style={
        'sim': {'color': 'k', 'linestyle': '-', 'alpha': 0.5},
        'mrp': {'color': 'b', 'linestyle': '--', 'alpha': 0.5},
        'quat': {'color': 'g', 'linestyle': ':', 'alpha': 0.5},
        'mekf': {'color': 'r', 'linestyle': '-.', 'alpha': 0.5},
        'default': {'color': 'c', 'linestyle': '-.', 'alpha': 0.5}
        }, fig_dir=results_dir)
    return data


if __name__ == "__main__":
    data = test_sim()