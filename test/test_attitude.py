import os

from pyecca2.estimators.attitude.derivation import derivation
from pyecca2.estimators.attitude.launch import launch_monte_carlo_sim
from pyecca2.estimators.attitude.plot import plot


def test_derivation():
    eqs = derivation()
    print('eqs', eqs.keys())


def test_sim():
    script_dir = os.path.abspath(os.path.dirname(__file__))
    fig_dir = os.path.join(script_dir, 'results', 'attitude')
    data = launch_monte_carlo_sim({'n': 2, 'tf': 100})
    plot(data, fig_dir)


if __name__ == "__main__":
    test_sim()