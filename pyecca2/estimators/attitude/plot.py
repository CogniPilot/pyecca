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

    # topic names
    est_state_topics = [name + '_state' for name in est_names]
    est_status_topics = [name + '_status' for name in est_names]
    gt_state = ground_truth_name + '_state'

    # computer start/stop inex
    i_start = np.argmax(data[0]['time'] > t_start)
    if t_stop > 0:
        i_stop = np.argmax(data[0]['time'] > t_stop)
    else:
        i_stop = -1

    # create output directory
    os.makedirs(fig_dir, exist_ok=True)

    def compare_topics(title, xlabel, ylabel, topics, get_data, *args, **kwargs):
        p = {
            'close_fig': True,
            'show': False
        }
        for k in p.keys():
            if k in kwargs.keys():
                p[k] = kwargs.pop(k)

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
                    for series_label, series in get_data(d, topic):
                        h = plt.plot(d['time'], series,
                                            *args, **style, **kwargs)[0]
                        if i == 0:
                            handles.append(h)
                            labels.append(series_label)
                except ValueError as e:
                    print(e)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.grid(True)
        plt.tight_layout()
        if len(handles) > 1:
            plt.legend(
                handles, labels, loc='best')
        plt.savefig(os.path.join(fig_dir, file_name))
        if p['show']:
            plt.show()
        if p['close_fig']:
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

    def compare_error_with_cov(title, xlabel, ylabel, topics, get_error, get_std, *args, **kwargs):
        p = {
            'close_fig': True,
            'show': False
        }
        for k in p.keys():
            if k in kwargs.keys():
                p[k] = kwargs.pop(k)

        file_name = title.replace(' ', '_').lower()
        fig = plt.figure()
        handles = []
        labels = []
        for i, d in enumerate(data):
            d = d[i_start:i_stop]
            for est in est_names:
                e = get_error(d, est)
                s = get_std(d, est)
                std_style = dict(est_style[est])
                std_style['linewidth'] = 1
                h1 = plt.plot(d['time'], e, **est_style[est])[0]
                h2 = plt.plot(d['time'], s, **std_style)[0]
                plt.plot(d['time'], -s, **std_style)
                if i == 0:
                    handles.append(h1)
                    labels.append(est)
                    handles.append(h2)
                    labels.append(est + ' std')
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.legend(handles, labels, loc='best')
        plt.grid()
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, file_name))
        if p['show']:
            plt.show()
        if p['close_fig']:
            plt.close(fig)

    # actual plotting starts here

    compare_topics(
        'quaternion normal error', 'time, sec', 'normal error', est_state_topics,
        lambda data, topic: [
            (topic, np.abs(1 - np.linalg.norm(data[topic]['q'], axis=1)))
        ])

    compare_topics(
        'prediction cpu usage', 'time, sec', 'cpu time, usec', est_status_topics,
        lambda data, topic: [
            (topic, 1e6 * data[topic]['cpu_predict'])
        ])

    compare_topics(
        'mag correct cpu usage', 'time, sec', 'cpu time, usec', est_status_topics,
        lambda data, topic: [
            (topic, 1e6 * data[topic]['cpu_mag'])
        ])

    compare_topics(
        'accel correct cpu usage', 'time, sec', 'cpu time, usec', est_status_topics,
        lambda data, topic: [
            (topic, 1e6 * data[topic]['cpu_accel'])
        ])

    compare_topics(
        'mag ret', 'time, sec', 'return code', est_status_topics,
        lambda data, topic: [
            (topic, data[topic]['mag_ret'])
        ])

    compare_topics(
        'mag innovation', 'time, sec', 'innovation, deg', est_status_topics,
        lambda data, topic: [
            (topic, np.rad2deg(data[topic]['r_mag']))
        ])

    compare_topics(
        'mag beta', 'time, sec', 'beta', est_status_topics,
        lambda data, topic: [
            (topic, data[topic]['beta_mag'])
        ])

    compare_topics(
        'accel ret', 'time, sec', 'return code', est_status_topics,
        lambda data, topic: [
            (topic, data[topic]['accel_ret'])
        ])

    compare_topics(
        'accel innovation', 'time, sec', 'innovation, m/s^2', est_status_topics,
        lambda data, topic: [
            (topic, data[topic]['r_accel'])
        ])

    compare_topics(
        'accel beta', 'time, sec', 'beta', est_status_topics,
        lambda data, topic: [
            (topic, data[topic]['beta_accel'])
        ])

    compare_topics(
        'rotation error', 'time, sec', 'error, deg', est_state_topics,
        lambda data, topic: [
            (topic, compare_rot_error(data[topic]['q'], data[gt_state]['q']))
        ])

    compare_error_with_cov(
        'rotation error', 'time, sec', 'error, deg', est_state_topics,
        get_error=lambda d, est: np.rad2deg(compare_rot_error(d[est + '_state']['q'], d[gt_state]['q'])),
        get_std=lambda d, est: np.rad2deg(d[est + '_status']['W'][:, 0:3])
    )

    compare_topics(
        'rotation error norm', 'time, sec', 'error, deg', est_state_topics,
        lambda data, topic: [
            (topic, compare_rot_error_norm(data[topic]['q'], data[gt_state]['q']))
        ])

    compare_topics(
        'angular velocity', 'time, sec', 'angular velocity, deg/s',
        [gt_state],
        lambda data, topic: [
            (topic, np.rad2deg(data[topic]['omega']))
        ])

    compare_topics(
        'quaternion', 'time, sec', 'quaternion component',
        [gt_state] + est_state_topics,
        lambda data, topic: [
            (topic, data[topic]['q'])
        ])

    compare_topics(
        'modified rodrigues params', 'time, sec', 'mrp component',
        [gt_state] + est_state_topics,
        lambda data, topic: [
            (topic, data[topic]['r'])
        ])

    compare_topics(
        'bias', 'time, sec', 'bias, deg/min',
        [gt_state] +  est_state_topics,
        lambda data, topic: [
            (topic, 60 * np.rad2deg(data[topic]['b']))
        ])

    compare_error_with_cov(
        'bias error', 'time, sec', 'bias error, deg/min',
        est_state_topics,
        get_error=lambda d, est: 60*180/np.pi*(d[est + '_state']['b'] - d[gt_state]['b']),
        get_std=lambda d, est: 60*180/np.pi*(d[est + '_status']['W'][:, 3:6])
    )

    compare_topics(
        'estimation uncertainty', 'time, sec', 'std. dev.',
        est_status_topics,
        lambda data, topic: [
            (topic, data[topic]['W'][:, :3])
        ])

    compare_topics(
        'mag', 'time, sec', 'magnetometer, normalized',
        ['mag'],
        lambda data, topic: [
            (topic, data[topic]['mag'])
        ])

    compare_topics(
        'accel', 'time, sec', 'accelerometer, m/s^2',
        ['imu'],
        lambda data, topic: [
            (topic, data[topic]['accel'])
        ])

    compare_topics(
        'gyro', 'time, sec', 'gyro, rad/s',
        ['imu'],
        lambda data, topic: [
            (topic, data[topic]['gyro'])
        ])
