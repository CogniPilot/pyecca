import os

import matplotlib.pyplot as plt
import numpy as np

from pyecca2.estimators.attitude import algorithms

eqs = algorithms.eqs()

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

    # computer start/stop index
    i_start = np.argmax(data[0]['time'] > t_start)
    if t_stop > 0:
        i_stop = np.argmax(data[0]['time'] > t_stop)
        if i_stop <= i_start:
            i_stop = -1
    else:
        i_stop = -1

    # create output directories
    pdf_dir = os.path.join(fig_dir, 'pdf')
    os.makedirs(pdf_dir, exist_ok=True)

    svg_dir = os.path.join(fig_dir, 'svg')
    os.makedirs(svg_dir, exist_ok=True)

    def compare_topics(title, xlabel, ylabel, est_topics, get_data, *args, **kwargs):
        try:
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
                for est in est_topics:
                    label = est
                    if label in est_style:
                        style = est_style[label]
                    elif 'default' in est_style:
                        style = est_style['default']
                    else:
                        style = {}
                    try:
                        for series_label, series in get_data(d, est):
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
                    handles, labels, loc='best', ncol=len(est_names))
            plt.savefig(os.path.join(svg_dir, file_name + '.svg'))
            plt.savefig(os.path.join(pdf_dir, file_name + '.pdf'))
            if p['show']:
                plt.show()
            if p['close_fig']:
                plt.close(fig)
        except Exception as e:
            print(title, 'failed', e)

    def compare_rot_error(q1, q2):
        r = []
        for q1i, q2i in zip(q1, q2):
            if np.isnan(q1i[0]) or np.isnan(q2i[0]):
                ri = np.array([np.nan, np.nan, np.nan])
            else:
                ri = np.array(eqs['sim']['rotation_error'](q1i, q2i))[:, 0]
            r.append(ri)
        r = np.array(r)
        return r

    def compare_rot_error_norm(q1, q2):
        r = []
        for q1i, q2i in zip(q1, q2):
            if np.isnan(q1i[0]) or np.isnan(q2i[0]):
                ri = np.nan
            else:
                ri = np.linalg.norm((eqs['sim']['rotation_error'](q1i, q2i)))
            r.append(ri)
        r = np.array(r)
        return r

    def compare_error_with_cov(title, xlabel, ylabel, est_topics, get_error, get_std, *args, **kwargs):
        try:
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
                for est in est_topics:
                    e = get_error(d, est)
                    s = get_std(d, est)
                    std_style = dict(est_style[est])
                    std_style['linewidth'] = est_style[est]['linewidth'] + 1
                    h1 = plt.plot(d['time'], e, **est_style[est])[0]
                    h2 = plt.plot(d['time'], 2 * s, **std_style)[0]
                    plt.plot(d['time'], -2 * s, **std_style)
                    if i == 0:
                        handles.append(h1)
                        labels.append(est)
                        handles.append(h2)
                        labels.append(est + r" 2$\sigma$")
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            plt.title(title)
            plt.grid()
            plt.legend(handles, labels, loc='best', ncol=len(est_names))
            plt.tight_layout()
            plt.savefig(os.path.join(svg_dir, file_name + '.svg'))
            plt.savefig(os.path.join(pdf_dir, file_name + '.pdf'))
            if p['show']:
                plt.show()
            if p['close_fig']:
                plt.close(fig)
        except Exception as e:
            print(title, 'failed', e)

    # actual plotting starts here

    compare_topics(
        'quaternion normal error', 'time, sec', 'normal error', est_names,
        lambda d, est: [
            (est, np.abs(1 - np.linalg.norm(d[est + '_state']['q'], axis=1)))
        ])

    compare_topics(
        'prediction cpu time', 'time, sec', 'cpu time, usec', est_names,
        lambda d, est: [
            (est, 1e6 * d[est + '_status']['cpu_predict'])
        ])

    compare_topics(
        'mag correct cpu time', 'time, sec', 'cpu time, usec', est_names,
        lambda d, est: [
            (est, 1e6 * d[est + '_status']['cpu_mag'])
        ])

    compare_topics(
        'accel correct cpu time', 'time, sec', 'cpu time, usec', est_names,
        lambda d, est: [
            (est, 1e6 * d[est + '_status']['cpu_accel'])
        ])

    compare_topics(
        'mag ret', 'time, sec', 'return code', est_names,
        lambda d, est: [
            (est, d[est + '_status']['mag_ret'])
        ])

    compare_error_with_cov(
        'mag innovation', 'time, sec', 'innovation, deg', est_names,
        get_error=lambda d, est: np.rad2deg(d[est + '_status']['r_mag']),
        get_std=lambda d, est: np.rad2deg(d[est + '_status']['r_std_mag']),
    )

    compare_topics(
        'mag beta', 'time, sec', 'beta', est_names,
        lambda d, est: [
            (est, d[est + '_status']['beta_mag'])
        ])

    compare_topics(
        'accel ret', 'time, sec', 'return code', est_names,
        lambda d, est: [
            (est, d[est + '_status']['accel_ret'])
        ])

    compare_error_with_cov(
        'accel innovation', 'time, sec', 'innovation, m/s^2', est_names,
        get_error=lambda d, est: d[est + '_status']['r_accel'],
        get_std=lambda d, est: d[est + '_status']['r_std_accel'],
    )

    compare_topics(
        'accel beta', 'time, sec', 'beta', est_names,
        lambda d, est: [
            (est, d[est + '_status']['beta_accel'])
        ])

    compare_error_with_cov(
        'rotation error', 'time, sec', 'error, deg', est_names,
        get_error=lambda d, est: np.rad2deg(compare_rot_error(d[est + '_state']['q'], d[gt_state]['q'])),
        get_std=lambda d, est: np.rad2deg(d[est + '_status']['W'][:, 0:3])
    )

    compare_topics(
        'rotation error norm', 'time, sec', 'error, deg', est_names,
        lambda d, est: [
            (est, np.rad2deg(compare_rot_error_norm(d[est + '_state']['q'], d[gt_state]['q'])))
        ])

    compare_topics(
        'angular velocity', 'time, sec', 'angular velocity, deg/s', [ground_truth_name],
        lambda d, est: [
            (est, np.rad2deg(d[est + '_state']['omega']))
        ])

    compare_topics(
        'quaternion', 'time, sec', 'quaternion component', [ground_truth_name] + est_names,
        lambda d, est: [
            (est, d[est + '_state']['q'])
        ])

    compare_topics(
        'modified rodrigues params', 'time, sec', 'mrp component', [ground_truth_name] + est_names,
        lambda d, est: [
            (est, d[est + '_state']['r'])
        ])

    compare_topics(
        'bias', 'time, sec', 'bias, deg/min', [ground_truth_name] + est_names,
        lambda d, est: [
            (est, 60 * np.rad2deg(d[est + '_state']['b']))
        ])

    compare_error_with_cov(
        'bias error', 'time, sec', 'bias error, deg/min', est_names,
        get_error=lambda d, est: 60 * 180 / np.pi * (d[est + '_state']['b'] - d[gt_state]['b']),
        get_std=lambda d, est: 60 * 180 / np.pi * (d[est + '_status']['W'][:, 3:6])
    )

    compare_topics(
        'mag', 'time, sec', 'magnetometer, normalized', ['mag'],
        lambda d, topic: [
            ('mag', d[topic]['mag'])
        ])

    compare_topics(
        'accel', 'time, sec', 'accelerometer, m/s^2', ['imu'],
        lambda d, topic: [
            ('accel', d[topic]['accel'])
        ])

    compare_topics(
        'gyro', 'time, sec', 'gyro, rad/s', ['imu'],
        lambda d, topic: [
            ('gyro', d[topic]['gyro'])
        ])
