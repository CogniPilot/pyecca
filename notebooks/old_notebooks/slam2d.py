import matplotlib.pyplot as plt
import numpy as np
import casadi as ca


def g(x: np.array, u: np.array, dt: float):
    """
    Vehicle dynamics propagation.

    @param x: vehicle state
    @param u: vehicle input
    @param dt: time step
    """
    return x + u * dt


def data_association(x: np.array, z: np.array, landmarks: np.array):
    """
    Associates measurement with known landmarks using maximum likelihood

    @param x: state of vehicle
    @param z: measurement
    @param landmarks: map of known landmarks
    """
    dm = landmarks - x
    rng_pred = np.linalg.norm(dm, axis=1)
    bearing_pred = np.arctan2(dm[:, 1], dm[:, 0])
    z_error_list = np.array([rng_pred, bearing_pred]).T - np.array([z])
    # print(z_error_list)
    Q = np.array([[1, 0], [0, 1]])
    Q_I = np.linalg.inv(Q)
    J_list = []
    for z_error in z_error_list:
        J_list.append(z_error.T @ Q_I @ z_error)
    J_list = np.array(J_list)
    i = np.argmin(J_list)
    # print(i)
    return i


def h(x, m, noise=None):
    """
    Predicts the measurements of a landmark at a given state. Returns None
    if out of range.

    @param x: vehicle static
    @param m: landmark
    @param noise: bool to control noise
    """
    dm = m - x
    d = np.linalg.norm(dm)
    m_range = d
    m_bearing = np.arctan2(dm[1], dm[0])
    if noise is not None:
        m_bearing += (
            noise["bearing_std"] * (np.random.randn() - 0.5) / 0.5
        )  # Is the flat addition of a random variable appropriate or should we scale with magnitude of bearing? also we changed so that the bearing noise is positive or negative

        m_range += (
            d * noise["range_std"] * np.random.randn()
        )  # This may shrink large range to be below range_max. Need to fix
    return [m_range, m_bearing]


def measure_landmarks(x, landmarks, noise=None, range_max=4):
    """
    Predicts all measurements at a given state

    @param x: vehicle static
    @param landmarks: list of existing landmarks
    @param noise: bool to control noise
    """
    z_list = []
    for m in landmarks:
        z = h(x, m, noise=noise)
        if z[0] < range_max:
            z_list.append(z)
    return z_list


def measure_odom(x, x_prev, noise=None):
    dx = x - x_prev
    d = np.linalg.norm(dx)
    theta = np.arctan2(dx[1], dx[0])
    odom = dx
    if noise is not None:
        R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        odom += d * (
            noise["odom_std"] * np.random.randn(2)
            + R @ np.array([noise["odom_bx_bias"], noise["odom_by_bias"]])
        )  # right now this is only positive noise, only overestimates
    return list(odom)


def simulate(noise=None, plot=False):
    x = np.array([0, 0])
    x_prev = x
    dt = 2

    landmarks = np.array(
        [
            [0, 2],
            [4, 6],
            [9, 1],
            [9, 1.5],
            [9, 10],
        ]
    )

    hist = {"t": [], "x": [], "u": [], "odom": [], "z": [], "odom_pred": []}

    for i in range(10):
        t = dt * i

        # measure landmarks
        z_list = measure_landmarks(x, landmarks, noise=noise)

        # propagate
        u = np.array([np.cos(t / 10), np.sin(t / 10)])
        x = g(x_prev, u, dt)  # predict

        # odometry
        odom = measure_odom(x, x_prev, noise=noise)
        odom_pred = measure_odom(x, x_prev)

        x_prev = x

        hist["t"].append(t)
        hist["x"].append(x)
        hist["u"].append(u)
        hist["odom"].append(np.hstack([odom, i]))
        hist["odom_pred"].append(np.hstack([odom_pred, i]))

        for z in z_list:
            hist["z"].append(np.hstack([z, i]))

    for key in ["t", "x", "u", "z", "odom"]:
        hist[key] = np.array(hist[key])

    if plot:
        fig = plt.figure(1)
        plt.plot(landmarks[:, 0], landmarks[:, 1], "bo", label="landmarks")
        plt.plot(hist["x"][:, 0], hist["x"][:, 1], "r.", label="states", markersize=10)

        # plot odom
        x_odom = np.array([0, 0], dtype=float)
        x_odom_hist = [x_odom]
        for odom in hist["odom"]:
            x_odom = np.array(x_odom) + np.array(odom[:2])
            x_odom_hist.append(x_odom)
        x_odom_hist = np.array(x_odom_hist)
        plt.plot(x_odom_hist[:, 0], x_odom_hist[:, 1], "g.", linewidth=3, label="odom")

        # plot measurements
        for rng, bearing, xi in hist["z"]:
            xi = int(xi)
            x = x_odom_hist[xi, :]
            plt.arrow(
                x[0],
                x[1],
                rng * np.cos(bearing),
                rng * np.sin(bearing),
                width=0.1,
                length_includes_head=True,
            )

        plt.axis([0, 10, 0, 10])
        plt.grid()
        plt.legend()
        plt.axis("equal")

    return locals()


def J_graph_slam(hist, x_meas, landmarks):
    J = 0

    Q = np.eye(2)  # meas covariance  ##look into more realistice covariance
    R = np.eye(2)  # odom covariance
    R_I = np.linalg.inv(R)
    Q_I = np.linalg.inv(Q)

    n_x = len(hist["x"])
    odom_pred = hist["odom_pred"]
    for i in range(n_x):
        # compute odom cost
        u = hist["u"][i]
        odom = hist["odom"][i]
        e_x = np.array(odom[:2]) - np.array(odom_pred[i][:2])
        J += e_x.T @ R_I @ e_x

    n_m = len(hist["z"])

    for i in range(n_m):
        # compute measurement cost
        rng, brg, xi = hist["z"][i]
        z_i = np.array([rng, brg])
        c_i = data_association(
            x_meas[int(xi)], z_i, landmarks
        )  # this should use x predicted based off of our input
        z_i_predicted = h(
            x_meas[i], landmarks[c_i]
        )  # this should use x predicted based off of our input
        e_z = np.array(z_i) - np.array(z_i_predicted)
        J += e_z.T @ Q_I @ e_z
        return J


def build_cost(odom, z, assoc, n_x, n_l):
    """
    @param odom : [delta_x, delta_y, xi] vertically stacked  (xi is state index assoc. with odom)
    @param z : [rng, bearing, xi] vertically stacked
    @param assoc : [ li ] landmark  associations for z, vertically stacked
    @param n_x : number of states
    @param n_l : number of landmarks
    """
    # constants
    # -------------------------------------------------------------

    # covariance for measurement
    Q = ca.SX(2, 2)
    rng_std = 1
    bearing_std = 1
    Q[0, 0] = rng_std**2
    Q[1, 1] = bearing_std**2
    Q_I = ca.inv(Q)

    # covariance for odometry
    R = ca.SX(2, 2)
    odom_x_cov = 1
    odom_y_cov = 1
    R[0, 0] = odom_x_cov**2
    R[1, 1] = odom_y_cov**2
    R_I = ca.inv(R)

    # what we want to optimized
    # ---------------------------------

    # the position of the current landmarks
    l = ca.SX.sym("l", n_l, 2)

    # the state
    x = ca.SX.sym("x", n_x, 2)
    # compute cost
    # -------------------
    J = 0
    x_prev = ca.vertcat(
        *([x[0, :]] + [x[i, :] for i in range(n_x - 1)])
    )  # change sim definitions to make this cleaner.
    x_all = ca.vertcat(ca.SX.zeros(1, 2), x)
    # for each odometry measurement
    for i in range(odom.shape[0]):
        odom_pred = x_all[i + 1, :] - x_all[i, :]  # double check with Goppert
        e_x = ca.SX.zeros(1, 2)
        e_x[0] = odom[i, :2][0] - odom_pred[0]
        e_x[1] = odom[i, :2][1] - odom_pred[1]
        J += e_x @ R_I @ e_x.T

    # for each (rng, bearing) measurement
    for j in range(z.shape[0]):
        rng, bearing, xi = z[j, :]
        li = assoc[j]

        # predicted measurement
        z_pred = ca.SX(2, 1)
        dm = l[li, :] - x_all[xi, :]
        z_pred[0] = ca.norm_2(dm)  # range
        z_pred[1] = ca.arctan2(dm[1], dm[0])  # bearing

        # error
        e_z = z[j, :2] - z_pred
        # cost
        J += e_z.T @ Q_I @ e_z

    return ca.Function("f_J", [x, l], [J], ["x", "l"], ["J"]), J


def plot_me(sim):
    hist = sim["hist"]
    landmarks = sim["landmarks"]
    fig = plt.figure(1)
    plt.plot(landmarks[:, 0], landmarks[:, 1], "bo", label="landmarks")
    plt.plot(hist["x"][:, 0], hist["x"][:, 1], "r.", label="states", markersize=10)
    # plot odom
    x_odom = np.array([0, 0], dtype=float)
    x_odom_hist = [x_odom]
    for odom in hist["odom"]:
        x_odom = np.array(x_odom) + np.array(odom[:2])
        x_odom_hist.append(x_odom)
    x_odom_hist = np.array(x_odom_hist)
    plt.plot(x_odom_hist[:, 0], x_odom_hist[:, 1], "g.", linewidth=3, label="odom")
    # plot measurements
    for rng, bearing, xi in hist["z"]:
        xi = int(xi)
        x = x_odom_hist[xi, :]
        plt.arrow(
            x[0],
            x[1],
            rng * np.cos(bearing),
            rng * np.sin(bearing),
            width=0.1,
            length_includes_head=True,
        )

    plt.axis([0, 10, 0, 10])
    plt.grid()
    plt.legend()
    plt.axis("equal")
    return
