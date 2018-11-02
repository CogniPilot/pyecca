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
        self.std_mag = uros.Param(core, 'sim/std_mag', 1e-2, 'f8')
        self.std_accel = uros.Param(core, 'sim/std_accel', 1e-3, 'f8')
        self.std_gyro = uros.Param(core, 'sim/std_gyro', 1e-6, 'f8')
        self.sn_gyro_rw = uros.Param(core, 'sim/sn_gyro_rw', 1e-6, 'f8')
        self.dt_sim = uros.Param(core, 'sim/dt_sim', 1.0 / 400, 'f8')
        self.dt_mag = uros.Param(core, 'sim/dt_mag', 1.0 / 50, 'f8')
        self.dt_imu = uros.Param(core, 'sim/dt_imu', 1.0 / 200, 'f8')
        self.mag_decl = uros.Param(core, 'sim/mag_decl', 0, 'f8')
        self.mag_incl = uros.Param(core, 'sim/mag_incl', 0, 'f8')
        self.mag_str = uros.Param(core, 'sim/mag_str', 1e-1, 'f8')
        self.g = uros.Param(core, 'sim/g', 9.8, 'f8')
        self.enable_noise = uros.Param(core, 'sim/enable_noise', False, '?')

        # msgs
        self.msg_sim_state = msgs.VehicleState()
        self.msg_imu = msgs.Imu()
        self.msg_mag = msgs.Mag()

        # misc
        self.t_last_sim = 0
        self.t_last_imu = 0
        self.t_last_mag = 0
        self.param_list = [self.std_mag, self.std_accel, self.std_gyro, self.sn_gyro_rw,
                           self.dt_sim, self.dt_mag, self.dt_imu,
                           self.mag_decl, self.mag_incl, self.mag_str,
                           self.g, self.enable_noise]
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
        omega_b = np.random.randn(3)
        omega_b = 20*omega_b/np.linalg.norm(omega_b)

        eps = 1e-7

        while True:

            # time
            t = self.core.now

            # compute dt
            dt = t - self.t_last_sim
            self.t_last_sim = t

            # propagate
            w_gyro_rw = self.randn(3)
            if t != 0:
                x = self.eqs['sim']['simulate'](
                    t, x, omega_b, self.sn_gyro_rw.get(), w_gyro_rw, dt)

            # measure and publish accel/gyro
            if t == 0 or t - self.t_last_imu >= self.dt_imu.get() - eps:
                self.t_last_imu = t

                # publish sim state
                q, b_g = self.eqs['sim']['get_state'](x)
                self.msg_sim_state.data['time'] = t
                self.msg_sim_state.data['q'] = q.T
                self.msg_sim_state.data['b'] = b_g.T
                self.msg_sim_state.data['omega'] = omega_b.T
                self.pub_sim.publish(self.msg_sim_state)

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
            if t == 0 or t - self.t_last_mag >= self.dt_mag.get() - eps:
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
