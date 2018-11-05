import os

import matplotlib.pyplot as plt
import numpy as np

from .derivation import derive_equations

eqs = derive_equations()

plt.rcParams['lines.linewidth'] = 2
plt.rcParams['lines.markersize'] = 7
plt.rcParams['lines.markeredgewidth'] = 1


def plot(data, ground_truth_name, est_names, est_style, fig_dir,
         t_start=0, t_stop=-1, show=False):
    plt.close('all')

    i_start = np.argmax(data[0]['time'] > t_start)
    if t_stop > 0:
        i_stop = np.argmax(data[0]['time'] > t_stop)
    else:
        i_stop = -1

    gt_state = ground_truth_name + '_state'

    os.makedirs(fig_dir, exist_ok=True)

    def compare_topics(title, xlabel, ylabel, topics, get_data, *args, **kwargs):
        file_name = title.replace(' ', '_').lower()
        fig = plt.figure()
        handles = []
        labels = []
        for i, d in enumerate(data):
            d = d[i_start:i_stop]
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
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.grid(True)
        plt.tight_layout()
        plt.legend(
            handles, labels, loc='best')
        plt.savefig(os.path.join(fig_dir, file_name))
        if show:
            plt.show()
        plt.close(fig)

    def compare_rot_error(q1, q2):
        r = []
        for q1i, q2i in zip(q1, q2):
            if np.isnan(q1i[0]) or np.isnan(q2i[0]):
                ri =  np.array([np.nan, np.nan, np.nan])
            else:
                ri = np.array(eqs['sim']['rotation_error'](q1i, q2i))[:, 0]
            r.append(ri)
        r = np.rad2deg(np.array(r))
        return r

    def compare_rot_error_norm(q1, q2):
        r = []
        for q1i, q2i in zip(q1, q2):
            if np.isnan(q1i[0]) or np.isnan(q2i[0]):
                ri = np.nan
            else:
                ri = np.linalg.norm((eqs['sim']['rotation_error'](q1i, q2i)))
            r.append(ri)
        r = np.rad2deg(np.array(r))
        return r

    est_state_topics = [name + '_state' for name in est_names]
    est_status_topics = [name + '_status' for name in est_names]

    compare_topics(
        'quaternion normal error', 'time, sec', 'normal error', est_state_topics,
        lambda data, topic: np.abs(1 - np.linalg.norm(data[topic]['q'], axis=1)))

    compare_topics(
        'prediction cpu usage', 'time, sec', 'cpu time, usec', est_status_topics,
        lambda data, topic: 1e6 * data[topic]['cpu_predict'])

    compare_topics(
        'mag correct cpu usage', 'time, sec', 'cpu time, usec', est_status_topics,
        lambda data, topic: 1e6 * data[topic]['cpu_mag'])

    compare_topics(
        'accel correct cpu usage', 'time, sec', 'cpu time, usec', est_status_topics,
        lambda data, topic: 1e6 * data[topic]['cpu_accel'])

    compare_topics(
        'mag ret', 'time, sec', 'return code', est_status_topics,
        lambda data, topic: data[topic]['mag_ret'])

    compare_topics(
        'mag innovation', 'time, sec', 'innovation, deg', est_status_topics,
        lambda data, topic: np.rad2deg(data[topic]['r_mag']))

    compare_topics(
        'mag beta', 'time, sec', 'beta', est_status_topics,
        lambda data, topic: data[topic]['beta_mag'])

    compare_topics(
        'accel ret', 'time, sec', 'return code', est_status_topics,
        lambda data, topic: data[topic]['accel_ret'])

    compare_topics(
        'accel innovation', 'time, sec', 'innovation, m/s^2', est_status_topics,
        lambda data, topic: data[topic]['r_accel'])

    compare_topics(
        'accel beta', 'time, sec', 'beta', est_status_topics,
        lambda data, topic: data[topic]['beta_accel'])

    compare_topics(
        'rotation error', 'time, sec', 'error, deg', est_state_topics,
        lambda data, topic: compare_rot_error(data[topic]['q'], data[gt_state]['q']))

    compare_topics(
        'rotation error norm', 'time, sec', 'error, deg', est_state_topics,
        lambda data, topic: compare_rot_error_norm(data[topic]['q'], data[gt_state]['q']))

    compare_topics(
        'angular velocity', 'time, sec', 'angular velocity, deg/s',
        [gt_state],
        lambda data, topic: np.rad2deg(data[topic]['omega']))

    compare_topics(
        'quaternion', 'time, sec', 'quaternion component',
        [gt_state] + est_state_topics,
        lambda data, topic: data[topic]['q'])

    compare_topics(
        'modified rodrigues params', 'time, sec', 'mrp component',
        [gt_state] + est_state_topics,
        lambda data, topic: data[topic]['r'])

    compare_topics(
        'bias', 'time, sec', 'bias, deg/min',
        [gt_state] +  est_state_topics,
        lambda data, topic: 60 * np.rad2deg(data[topic]['b']))

    compare_topics(
        'bias error', 'time, sec', 'bias error, deg/min',
        est_state_topics,
        lambda data, topic: np.rad2deg(data[topic]['b'] - data[gt_state]['b']))

    compare_topics(
        'estimation uncertainty', 'time, sec', 'std. dev.',
        est_status_topics,
        lambda data, topic: data[topic]['W'][:, :3])

    compare_topics(
        'mag', 'time, sec', 'magnetometer, normalized',
        ['mag'],
        lambda data, topic: data[topic]['mag'])

    compare_topics(
        'accel', 'time, sec', 'accelerometer, m/s^2',
        ['imu'],
        lambda data, topic: data[topic]['accel'])

    compare_topics(
        'gyro', 'time, sec', 'gyro, rad/s',
        ['imu'],
        lambda data, topic: data[topic]['gyro'])
