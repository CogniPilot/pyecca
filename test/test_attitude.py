import os
import pickle
import numpy as np
import time

from pyecca.estimators.attitude import algorithms
from pyecca.estimators.attitude import launch
from pyecca.estimators.attitude.plot import plot


script_dir = os.path.abspath(os.path.dirname(__file__))
results_dir = os.path.join(script_dir, "results", "attitude")
data_dir = os.path.join(script_dir, "data")

alpha = 0.5
est_style = {
    "sim": {"color": "k", "linestyle": "-.", "linewidth": 1, "alpha": alpha},
    "ground_truth": {"color": "k", "linestyle": "-.", "linewidth": 1, "alpha": alpha},
    "mrp": {"color": "b", "linestyle": "--", "linewidth": 1, "alpha": alpha},
    "quat": {"color": "g", "linestyle": "-.", "linewidth": 1, "alpha": alpha},
    "mekf": {"color": "r", "linestyle": "-.", "linewidth": 1, "alpha": alpha},
    "replay": {"color": "c", "linestyle": "-.", "linewidth": 1, "alpha": alpha},
    "default": {"color": "m", "linestyle": "-.", "linewidth": 1, "alpha": alpha},
}


def test_derivation():
    eqs = algorithms.eqs(results_dir=results_dir)


def test_sim():
    params = {
        "n_monte_carlo": 8,
        "t0": 0,
        "tf": 10,
        "initialize": False,
        "estimators": ["mrp"],
        "x0": np.array([0.1, 0.2, 0.3, 0, 0.07, 0.02, -0.07]),
        "params": {
            "sim/dt_sim": 1.0 / 400,
            "sim/dt_imu": 1.0 / 200,
            "sim/dt_mag": 1.0 / 50,
            "sim/mag_incl": 0.3,
            # 'mekf/dt_min_mag': 1,
            # 'mekf/dt_min_accel': 0.5,
            # 'mrp/dt_min_mag': 1.0 / 50,
            # 'mrp/dt_min_accel': 1.0 / 200,
            # 'quat/dt_min_mag': 1.0 / 50,
            # 'quat/dt_min_accel': 1.0 / 200,
            "logger/dt": 1.0 / 200,
            "sim/enable_noise": True,
        },
    }
    start = time.perf_counter()
    data = launch.launch_monte_carlo_sim(params)
    elapsed = time.perf_counter() - start
    print("\n\nsimulation complete")
    print("-" * 30)
    print("cpu time\t\t:", np.round(elapsed, 2))
    print("tf\t\t\t\t:", params["tf"])
    print("n monte carlo\t:", params["n_monte_carlo"])
    print("n estimators\t:", len(params["estimators"]))
    print(
        "speed ratio\t\t:",
        np.round(
            len(params["estimators"])
            * params["n_monte_carlo"]
            * params["tf"]
            / elapsed,
            2,
        ),
    )

    data_path = os.path.join(results_dir, "data.pkl")

    os.makedirs(results_dir, exist_ok=True)
    with open(data_path, "wb") as f:
        pickle.dump(data, f)

    with open(data_path, "rb") as f:
        data = pickle.load(f)

    plot(
        data,
        ground_truth_name="sim",
        est_names=params["estimators"],
        est_style=est_style,
        fig_dir=os.path.join(results_dir, "sim"),
        t_start=params["t0"] + 0.1,
        t_stop=params["tf"],
        show=False,
    )


def test_generate_code():
    eqs = algorithms.eqs()
    algorithms.generate_code(eqs, os.path.join(results_dir, "code"))


def test_replay():
    params = {
        "t0": 0,
        "tf": 20,
        "initialize": False,
        "estimators": ["mrp"],
        "replay_log_file": os.path.join(
            data_dir, "9db86e4e-0910-41f8-9e0b-d16352ac8e90.ulg"
        ),
        "params": {},
    }
    data = launch.launch_replay(params)
    data_path = os.path.join(results_dir, "data.pkl")
    os.makedirs(results_dir, exist_ok=True)
    with open(data_path, "wb") as f:
        pickle.dump(data, f)

    with open(data_path, "rb") as f:
        data = pickle.load(f)

    plot(
        [data],
        ground_truth_name="ground_truth",
        est_names=params["estimators"] + ["log"],
        est_style=est_style,
        fig_dir=os.path.join(results_dir, "replay"),
        t_start=params["t0"] + 1,
        t_stop=params["tf"],
        show=False,
    )
