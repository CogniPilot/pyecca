import numpy as np
from tqdm import tqdm  # for progress bar, can install via pip
import scipy.integrate
import matplotlib.pyplot as plt


def sim(func, params):
    p = params
    
    # all data will be stored in this dictionary, 
    hist = {}

    # noise matrices
    dt_mag = p['dt']*p['mod_mag']
    dt_accel = p['dt']*p['mod_accel']
    R_accel = np.diag([1, 1])*p['accel_sqrt_N']**2/dt_accel
    R_mag = np.diag([1])*p['mag_sqrt_N']**2/dt_mag
    P = np.zeros((6, 6))
    for i in range(3):
        P[i, i] = p['std0_att']**2
        P[i+3, i+3] = p['std0_bias']**2
    Q = np.zeros((6, 6))
    for i in range(3):
        Q[i, i] = p['std_w_att']**2/p['dt']
        Q[i+3, i+3] = p['std_w_bias']**2/p['dt']

    # states
    x = p['x0']
    bg = p['f_bg'](0)
    bgh = np.zeros(3)
    q = np.reshape(func['mrp_to_quat'](x), -1)
    xh = np.array([0, 0, 0])
    qh = np.reshape(func['mrp_to_quat'](xh), -1)
    y_accel = np.reshape(func['measure_g'](x), -1)
    y_mag = np.reshape(func['measure_hdg'](x), -1)
    yh_accel = np.array([0, 0, 0])
    yh_mag = np.array([0, 0, 0])
    eta_R = np.array([0, 0, 0])
    shadow = 0 # need to track shadow state to give a consistent quaternion
    shadowh = 0 # need to track shadow state to give a consistent quaternion
    x_delayed = [] # simulate measurement delay
    i = 0
    
    def get_x_delayed(periods):
        if periods < len(x):
            return x_delayed[periods]
        else:
            return x_delayed[-1]

    def handle_shadow(x, s, q):
        if np.linalg.norm(x) > 1:
            x = np.reshape(func['mrp_shadow'](x), -1)
            s = not s
        q = func['mrp_to_quat'](x)
        if s:
            q *= -1
        return x, s, q

    t_vals = np.arange(0, p['tf'], p['dt'])

    # tqdm creates a progress bar from the range
    for t in tqdm(t_vals):
        i += 1

        # get gyro bias and rotation rate
        bg = p['f_bg'](t)
        omega = p['f_omega'](t)
        
        # simulate the actual motion of the rigid body
        res = scipy.integrate.solve_ivp(
            fun=func['dynamics'](omega), t_span=[t, t + p['dt']],
            y0=x)
        x = np.array(res['y'][:, -1])
        x, shadow, q = handle_shadow(x, shadow, q)
        omega_meas = omega + bg

        # predict the motion of the rigid body
        resh = scipy.integrate.solve_ivp(
            fun=func['dynamics'](omega_meas - bgh), t_span=[t, t + p['dt']],
            y0=xh)
        xh = np.array(resh['y'][:, -1])
        xh, shadowh, qh = handle_shadow(xh, shadowh, qh)
        
        # propagate the uncertainty
        P += np.array(func['dP'](np.hstack([xh, bgh]), P, Q)*p['dt'])

        # correction for accel
        if i % p['mod_accel'] == 0:
            w = np.random.randn(2)*np.sqrt(np.diag(R_accel))
            y_accel = func['f_noise_SO3']([w[0], w[1], 0], func['measure_g'](get_x_delayed(p['accel_delay_periods'])))
            yh_accel = func['measure_g'](xh)
            x_accel, P_accel = func['correct_accel'](np.hstack([xh, bgh]), P, y_accel, R_accel)
            xh = np.reshape(x_accel[0:3], -1)
            bgh = np.reshape(x_accel[3:6], -1)
            P = np.array(P_accel)
            xh, shadowh, qh = handle_shadow(xh, shadowh, qh)

        # correction for mag
        if i % p['mod_mag'] == 0:
            # simulate measurement
            w = np.random.randn(1)*np.sqrt(np.diag(R_mag))
            y_mag = func['f_noise_SO3']([0, 0, w], func['measure_hdg'](get_x_delayed(p['mag_delay_periods'])))
            yh_mag = func['measure_hdg'](xh)
            B_n = [1, 0, 0]
            x_mag, P_mag = func['correct_mag'](np.hstack([xh, bgh]), P, y_mag, B_n, R_mag)
            xh = np.reshape(x_mag[0:3], -1)
            bgh = np.reshape(x_mag[3:6], -1)
            P = np.array(P_mag)
            xh, shadowh, qh = handle_shadow(xh, shadowh, qh)

        data = ({
            'bg': bg,
            'bgh': bgh,
            'euler': func['quat_to_euler'](q),
            'eulerh': func['quat_to_euler'](qh),
            'q': q,
            'qh': qh,
            'shadow': shadow,
            'shadowh': shadowh,
            't': res['t'][-1],
            'x': x,
            'xh': xh,
            'xi': func['xi_R'](x, xh),
            'y_accel': y_accel,
            'y_mag': y_mag,
            'yh_accel': yh_accel,
            'yh_mag': yh_mag,
            'std': np.sqrt(np.reshape(np.diag(P), -1)),
        })
        for key in data.keys():
            if key not in hist.keys():
                hist[key] = []
            hist[key].append(np.reshape(data[key], -1))
        t += p['dt']
        x_delayed.insert(0, x)
        while (len(x_delayed) > (p['max_delay_periods'] + 1)):
            x_delayed.pop()

    for k in hist.keys():
        hist[k] = np.array(hist[k])
    return hist

def analyze_hist(hist, t=3):
    ti = int(t/(hist['t'][1] - hist['t'][0]))
    if ti < len(hist['t']):
        mean = list(np.rad2deg(np.mean(hist['xi'][ti:], 0)))
        std = list(np.rad2deg(np.std(hist['xi'][ti:], 0)))
        print('error statistics after {:0.0f} seconds'.format(t))
        print('mean (deg)\t: {:10.4f} roll, {:10.4f} pitch, {:10.4f} yaw'.format(*mean))
        print('std  (deg)\t: {:10.4f} roll, {:10.4f} pitch, {:10.4f} yaw'.format(*std))
        
def plot_hist(hist, figsize = (15, 10)):
    plt.figure(figsize=figsize)
    tf = hist['t'][-1]
    
    plt.subplot(411)
    plt.plot(hist['t'], hist['x'], 'r')
    plt.plot(hist['t'], hist['xh'], 'k')
    plt.grid()
    #plt.xlabel('t, sec')
    plt.ylabel('mrp')
    plt.gca().set_ylim(-1, 1)
    plt.gca().set_xlim(0, tf)
    plt.title('attitude representations')

    plt.subplot(412)
    plt.plot(hist['t'], hist['shadow'], 'r')
    plt.plot(hist['t'], hist['shadowh'], 'k')
    plt.ylabel('mrp shadow id')
    plt.gca().set_xlim(0, tf)
    plt.grid()
    plt.xlabel('t, sec')
    
    plt.subplot(413)
    plt.plot(hist['t'], hist['q'], 'r')
    plt.plot(hist['t'], hist['qh'], 'k')
    #plt.xlabel('t, sec')
    plt.ylabel('q')
    plt.grid()
    plt.gca().set_ylim(-1, 1)
    plt.gca().set_xlim(0, tf)

    plt.subplot(414)
    plt.plot(hist['t'], np.rad2deg(hist['euler']), 'r')
    plt.plot(hist['t'], np.rad2deg(hist['eulerh']), 'k')
    plt.ylabel('euler, deg')
    plt.grid()
    plt.gca().set_ylim(-200, 200)
    plt.gca().set_xlim(0, tf)
    plt.xlabel('t, sec')

    plt.figure(figsize=figsize)
    plt.subplot(211)
    plt.plot(hist['t'], hist['y_accel'], 'r')
    plt.plot(hist['t'], hist['yh_accel'], 'k')
    plt.ylabel('accel., norm.')
    plt.gca().set_xlim(0, tf)
    plt.grid()
    plt.title('measurements')

    plt.subplot(212)
    plt.plot(hist['t'], hist['y_mag'], 'r')
    plt.plot(hist['t'], hist['yh_mag'], 'k')
    plt.xlabel('t, sec')
    plt.ylabel('mag., norm.')
    plt.gca().set_xlim(0, tf)
    plt.grid()

    plt.figure(figsize=figsize)

    plt.subplot(211)
    plt.title('estimates')
    h_bg = plt.plot(hist['t'], np.rad2deg(hist['bg']), 'r--')
    std_b =  hist['std'][:, 3:6]
    h_sig = plt.plot(hist['t'], np.rad2deg(hist['bgh'] + 3*std_b), 'g-.')
    plt.plot(hist['t'], np.rad2deg(hist['bgh'] -3*std_b), 'g-.')
    h_bgh = plt.plot(hist['t'], np.rad2deg(hist['bgh']), 'k')
    plt.ylabel('gyro bias, deg/s')
    plt.gca().set_xlim(0, tf)
    plt.gca().set_ylim(-40, 40)
    plt.grid()
    plt.legend([h_bg[0], h_bgh[0], h_sig[0]],
               ['$b_g$', '$\hat{b}_g$', '3 $\sigma$'],
               loc='lower right', ncol=3)

    plt.subplot(212)
    std_r =  hist['std'][:, 0:3]
    h_sig = plt.plot(hist['t'], np.rad2deg(3*std_r), 'g-.')
    plt.plot(hist['t'], np.rad2deg(-3*std_r), 'g-.')
    h_eta = plt.plot(hist['t'], np.rad2deg(hist['xi']), 'k')
    plt.gca().set_ylim(-10, 10)
    plt.ylabel('rotation error, deg')
    plt.gca().set_xlim(0, tf)
    plt.xlabel('t, sec')
    plt.grid()
    plt.legend([h_eta[0], h_sig[0]], ['$\\xi$', '3 $\sigma$'], loc='lower right')

    plt.show()