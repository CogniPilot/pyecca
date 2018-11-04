import os

import matplotlib.pyplot as plt
import numpy as np

from .derivation import derivation

eqs = derivation()

plt.rcParams['lines.linewidth'] = 2
plt.rcParams['lines.markersize'] = 7
plt.rcParams['lines.markeredgewidth'] = 1


def plot(data, ground_truth_name, est_names, est_style, fig_dir):
    plt.close('all')

    gt_state = ground_truth_name + '_state'

    os.makedirs(fig_dir, exist_ok=True)

    def compare_topics(topics, get_data, *args, **kwargs):
        handles = []
        labels = []
        for i, d in enumerate(data):
            for topic in topics:
                label = topic.split('_')[0]
                if label in est_style:
                    style = est_style[label]
                elif 'default' in est_style:
                    style = est_style['default']
                else:
                    style = {}
                try:
                    h = plt.plot(d['time'], get_data(d, topic),
                                        *args, **style, **kwargs)[0]
                    if i == 0:
                        handles.append(h)
                        labels.append(label)
                except ValueError as e:
                    print(e)
        plt.legend(
            handles, labels, loc='best')

    def plot_handling(title, xlabel, ylabel, file_name):
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, file_name))

    def compare_rot_error(q1, q2):
        r = []
        for q1i, q2i in zip(q1, q2):
            ri = np.array(eqs['sim']['rotation_error'](q1i, q2i))[:, 0]
            r.append(ri)
        r = np.rad2deg(np.array(r))
        return r

    def compare_rot_error_norm(q1, q2):
        r = []
        for q1i, q2i in zip(q1, q2):
            ri = np.linalg.norm((eqs['sim']['rotation_error'](q1i, q2i)))
            r.append(ri)
        r = np.rad2deg(np.array(r))
        return r


    est_state_topics = [name + '_state' for name in est_names]
    est_status_topics = [name + '_status' for name in est_names]

    plt.figure()
    compare_topics(est_state_topics,
                   lambda data, topic: np.abs(1 - np.linalg.norm(data[topic]['q'], axis=1)))
    plot_handling('quaternion normal error', 'time, sec', 'normal error', 'quat_normal.png')

    plt.figure()
    compare_topics(est_status_topics,
                   lambda data, topic: 1e6 * data[topic]['elapsed'])
    plot_handling('cpu time', 'time, sec', 'cpu time, usec', 'cpu_time.png')

    plt.figure()
    compare_topics(est_status_topics,
                   lambda data, topic: data[topic]['mag_ret'])
    plot_handling('mag ret', 'time, sec', 'return code', 'mag_ret.png')

    plt.figure()
    compare_topics(est_status_topics,
                   lambda data, topic: np.rad2deg(data[topic]['r_mag']))
    plot_handling('mag innovation', 'time, sec', 'innovation, deg', 'mag_innov.png')

    plt.figure()
    compare_topics(est_status_topics,
                   lambda data, topic: data[topic]['beta_mag'])
    plot_handling('mag beta', 'time, sec', 'beta', 'mag_beta.png')

    plt.figure()
    compare_topics(est_status_topics,
                   lambda data, topic: data[topic]['accel_ret'])
    plot_handling('accel ret', 'time, sec', 'return code', 'accel_ret.png')

    plt.figure()
    compare_topics(est_status_topics,
                   lambda data, topic: data[topic]['r_accel'])
    plot_handling('accel innovation', 'time, sec', 'innovation, m/s^2', 'accel_innov.png')

    plt.figure()
    compare_topics(est_status_topics,
                   lambda data, topic: data[topic]['beta_accel'])
    plot_handling('accel beta', 'time, sec', 'beta', 'accel_beta.png')


    plt.figure()
    compare_topics(est_state_topics,
                   lambda data, topic: compare_rot_error(data[topic]['q'], data[gt_state]['q']))
    plot_handling('rotation error', 'time, sec', 'error, deg', 'rotation_error.png')

    plt.figure()
    compare_topics(est_state_topics,
                   lambda data, topic: compare_rot_error_norm(data[topic]['q'], data[gt_state]['q']))
    plot_handling('rotation error norm', 'time, sec', 'error, deg', 'rotation_error_norm.png')

    plt.figure()
    for d in data:
        plt.plot(d['time'], np.rad2deg(d[gt_state]['omega']))
    plot_handling('angular velocity', 'time, sec', 'angular velocity, deg/s', 'angular_velocity.png')

    plt.figure()
    compare_topics([gt_state] + est_state_topics,
                   lambda data, topic: data[topic]['q'])
    plot_handling('quaternion', 'time, sec', 'quaternion component', 'quaternion.png')

    plt.figure()
    compare_topics([gt_state] + est_state_topics,
                   lambda data, topic: data[topic]['r'])
    plot_handling('modified rodrigues params', 'time, sec', 'mrp component', 'mrp.png')

    plt.figure()
    compare_topics([gt_state] +  est_state_topics,
                   lambda data, topic: 60 * np.rad2deg(data[topic]['b']))
    plot_handling('bias', 'time, sec', 'bias, deg/min', 'bias.png')

    plt.figure()
    compare_topics(est_state_topics,
                   lambda data, topic: np.rad2deg(data[topic]['b'] - data[gt_state]['b']))
    plot_handling('bias error', 'time, sec', 'bias error, deg/min', 'bias_error.png')

    plt.figure()
    compare_topics(est_status_topics,
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
