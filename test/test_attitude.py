import os
import sys

script_dir = os.path.abspath(os.path.dirname(__file__))
root_dir = os.path.abspath(os.path.join(script_dir, os.path.pardir, os.path.pardir, os.path.pardir))
sys.path.insert(0, root_dir)


from pyecca2.estimators.attitude.launch import launch_monte_carlo_sim
from pyecca2.estimators.attitude.plot import plot


def test_sim():
    fig_dir = os.path.join(script_dir, 'results', 'attitude')
    data = launch_monte_carlo_sim(n=1)
    plot(data, fig_dir)


if __name__ == "__main__":
    test_sim()