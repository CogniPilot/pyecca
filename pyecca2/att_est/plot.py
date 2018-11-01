import os

import matplotlib.pyplot as plt
import numpy as np

import casadi as ca

import pyecca2.rotation as rot

est_style = {
    'true': {'color': 'k', 'linewidth': 2, 'linestyle': '-', 'alpha': 0.5},
    'mrp': {'color': 'b', 'linewidth': 2, 'linestyle': '-.', 'alpha': 0.5},
    'quat': {'color': 'g', 'linewidth': 2, 'linestyle': ':', 'alpha': 0.5},
    'mekf': {'color': 'r', 'linewidth': 2, 'linestyle': '--', 'alpha': 0.5},
}

label_map = {
    'sim_state': 'true',
    'est1_state': 'mekf',
    'est2_state': 'quat',
    'est3_state': 'mrp',
    'est1_status': 'mekf',
    'est2_status': 'quat',
    'est3_status': 'mrp',
}


def plot(data, fig_dir):
    plt.close('all')

    if not os.path.exists(fig_dir):
        os.mkdir(fig_dir)

    def compare_topics(topics, get_data, *args, **kwargs):
        h = {}
        for d in data:
            for topic in topics:
                label = label_map[topic]
                h[topic] = plt.plot(d['time'], get_data(d, topic),
                                    *args, **est_style[label], **kwargs)
        plt.legend(
            [v[0] for k, v in h.items()], [label_map[topic] for topic in topics],
            loc='best')

    def plot_handling(title, xlabel, ylabel, file_name):
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, file_name))

    plt.figure()
    compare_topics(['est1_state', 'est2_state', 'est3_state'],
                   lambda data, topic: np.linalg.norm(data[topic]['q'], axis=1) - 1)
    plot_handling('quaternion normal error', 'time, sec', 'normal error', 'quat_normal.png')

    plt.figure()
    compare_topics(['est1_status', 'est2_status', 'est3_status'],
                   lambda data, topic: 1e6 * data[topic]['elapsed'])
    plot_handling('cpu time', 'time, sec', 'cpu time, usec', 'cpu_time.png')

    plt.figure()

    def compare_rot_error(q1, q2):
        r = []
        for q1i, q2i in zip(q1, q2):
            q1i = rot.Quat(q1i)
            q2i = rot.Quat(q2i)
            dR = rot.SO3(rot.Dcm.from_quat(q1i.inv() * q2i))
            ri = np.linalg.norm(ca.DM(rot.SO3.log(dR)))
            r.append(ri)
        r = np.rad2deg(np.array(r))
        return r

    compare_topics(['est1_state', 'est2_state', 'est3_state'],
                   lambda data, topic: compare_rot_error(data[topic]['q'], data['sim_state']['q']))
    plot_handling('rotation error', 'time, sec', 'error, deg', 'rotation_error.png')

    plt.figure()
    for d in data:
        plt.plot(d['time'], np.rad2deg(d['sim_state']['omega']))
    plot_handling('angular velocity', 'time, sec', 'angular velocity, deg/s', 'angular_velocity.png')

    plt.figure()
    compare_topics(['sim_state', 'est1_state', 'est2_state', 'est3_state'],
                   lambda data, topic: data[topic]['q'])
    plot_handling('quaternion', 'time, sec', 'quaternion component', 'quaternion.png')

    plt.figure()
    compare_topics(['sim_state', 'est1_state', 'est2_state', 'est3_state'],
                   lambda data, topic: 3600 * np.rad2deg(data[topic]['b']))
    plot_handling('bias', 'time, sec', 'bias, deg/hr', 'bias.png')

    plt.figure()
    compare_topics(['est1_status', 'est2_status', 'est3_status'],
                   lambda data, topic: data[topic]['W'][:, :3])
    plot_handling('estimation uncertainty', 'time, sec', 'std. dev.', 'est_uncertainty.png')

    plt.figure()
    for d in data:
        plt.plot(d['time'], d['mag']['mag'])
    plot_handling('mag', 'time, sec', 'magnetometer, norm.', 'mag.png')

    plt.figure()
    for d in data:
        plt.plot(d['time'], d['imu']['accel'])
    plot_handling('accel', 'time, sec', 'accelerometer, m/s^2', 'accel.png')

    plt.figure()
    for d in data:
        plt.plot(d['time'], d['imu']['gyro'])
    plot_handling('gyro', 'time, sec', 'gyro, rad/s', 'gyro.png')

    plt.show()
