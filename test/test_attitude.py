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
    data = launch_monte_carlo_sim({
        'n_monte_carlo': 1,
        'tf': 10,
        'params': {
            'sim/dt_mag': 1.0/1,
            'logger/dt': 1.0/200,
            'sim/enable_noise': False
        }
    })

    with open(os.path.join(results_dir, 'data.pkl'), 'wb') as f:
        pickle.dump(data, f)

    plot(data, results_dir)
    return data


if __name__ == "__main__":
    data = test_sim()