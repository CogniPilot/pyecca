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
    data = launch_monte_carlo_sim({
        'n_monte_carlo': 1,
        'tf': tf,
        'params': {
            'sim/dt_sim': 1.0/400,
            'sim/dt_mag': 1.0/2,
            'logger/dt': tf/30,
            'sim/enable_noise': True
        }
    })

    with open(os.path.join(results_dir, 'data.pkl'), 'wb') as f:
        pickle.dump(data, f)

    plot(data, results_dir)
    return data


if __name__ == "__main__":
    data = test_sim()