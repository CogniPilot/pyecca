import numpy as np
import simpy

import pyecca2.msgs as msgs
import pyecca2.uros as uros


class Simulator:

    def __init__(self, core, eqs):
        self.core = core

        # publications
        self.pub_sim = uros.Publisher(core, 'sim_state', msgs.VehicleState)
        self.pub_imu = uros.Publisher(core, 'imu', msgs.Imu)
        self.pub_mag = uros.Publisher(core, 'mag', msgs.Mag)

        # subscriptions
        self.sub_params = uros.Subscriber(core, 'params', msgs.Params, self.params_callback)

        # parameters
        self.param_list = []

        def add_param(name, value, type):
            p = uros.Param(self.core, 'sim/' + name, value, type)
            self.param_list.append(p)
            return p

        self.std_mag = add_param('std_mag', 1e-3, 'f8')
        self.std_accel = add_param('std_accel', 1e-3, 'f8')
        self.std_gyro = add_param('std_gyro', 1e-6, 'f8')
        self.sn_gyro_rw = add_param('sn_gyro_rw', 1e-6, 'f8')
        self.dt_sim = add_param('dt_sim', 1.0 / 400, 'f8')
        self.dt_mag = add_param('dt_mag', 1.0 / 50, 'f8')
        self.dt_imu = add_param('dt_imu', 1.0 / 200, 'f8')
        self.mag_decl = add_param('mag_decl', 0, 'f8')
        self.mag_incl = add_param('mag_incl', 0, 'f8')
        self.mag_str = add_param('mag_str', 1e-1, 'f8')
        self.g = add_param('g', 9.8, 'f8')
        self.enable_noise = add_param('enable_noise', True, '?')

        # msgs
        self.msg_sim_state = msgs.VehicleState()
        self.msg_imu = msgs.Imu()
        self.msg_mag = msgs.Mag()

        # misc
        self.t_last_sim = 0
        self.t_last_imu = 0
        self.t_last_mag = 0

        self.eqs = eqs
        np.random.seed()
        simpy.Process(core, self.run())

    def params_callback(self, msg):
        for p in self.param_list:
            p.update()

    def randn(self, *args, **kwargs):
        return np.random.randn(*args) * self.enable_noise.get()

    def run(self):
        x = self.eqs['sim']['constants']()['x0']
        i = 0

        # true angular velocity, nav frame
        #omega_b = np.random.randn(3)
        #omega_b = 20*omega_b/np.linalg.norm(omega_b)
        #omega_b = np.array([3, 4, 5])

        eps = 1e-7

        while True:

            # time
            t = self.core.now

            omega_b = 0.1*np.array([
                1*(1 + np.sin(t*2*np.pi*1 + 1))/2,
                2*(1 + np.sin(t*2*np.pi*2 + 2))/2,
                3*(1 + np.sin(t*2*np.pi*3 + 3))/2
            ])

            # compute dt
            dt = t - self.t_last_sim
            self.t_last_sim = t

            # propagate
            w_gyro_rw = self.randn(3)
            if t != 0:
                x = self.eqs['sim']['simulate'](
                    t, x, omega_b, self.sn_gyro_rw.get(), w_gyro_rw, dt)

            # publish sim state
            q, r, b_g = self.eqs['sim']['get_state'](x)
            self.msg_sim_state.data['time'] = t
            self.msg_sim_state.data['q'] = q.T
            self.msg_sim_state.data['r'] = r.T
            self.msg_sim_state.data['b'] = b_g.T
            self.msg_sim_state.data['omega'] = omega_b.T
            self.pub_sim.publish(self.msg_sim_state)

            # measure and publish accel/gyro
            if t== 0 or t - self.t_last_imu >= self.dt_imu.get() - eps:
                self.t_last_imu = t

                # measure
                w_gyro = self.randn(3)
                w_accel = self.randn(3)
                y_gyro = self.eqs['sim']['measure_gyro'](
                    x, omega_b, self.std_gyro.get(), w_gyro).T
                y_accel = self.eqs['sim']['measure_accel'](
                    x, self.g.get(), self.std_accel.get(), w_accel).T

                # publish
                self.msg_imu.data['time'] = t
                self.msg_imu.data['gyro'] = y_gyro
                self.msg_imu.data['accel'] = y_accel
                self.pub_imu.publish(self.msg_imu)

            # measure and publish mag
            if t - self.t_last_mag >= self.dt_mag.get() - eps:
                self.t_last_mag = t

                # measure
                w_mag = self.randn(3)
                y_mag = self.eqs['sim']['measure_mag'](
                    x, self.mag_str.get(), self.mag_decl.get(), self.mag_incl.get(),
                    self.std_mag.get(), w_mag).T

                # publish
                self.msg_mag.data['time'] = t
                self.msg_mag.data['mag'] = y_mag
                self.pub_mag.publish(self.msg_mag)

            i += 1
            yield simpy.Timeout(self.core, self.dt_sim.get())
