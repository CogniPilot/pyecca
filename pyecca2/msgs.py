import numpy as np


def init_data(dtype):
    return np.zeros(1, dtype=dtype)[0]


time_type = 'f8'
float_type = 'f8'


class Imu:
    dtype = [
        ('time', time_type),  # timestamp
        ('gyro', float_type, 3),  # gyroscope measurement
        ('accel', float_type, 3),  # accelerometer measurement
    ]

    def __init__(self):
        self.data = init_data(self.dtype)


class Mag:
    dtype = [
        ('time', time_type),  # timestamp
        ('mag', float_type, 3),  # magnetometer measurement
    ]

    def __init__(self):
        self.data = init_data(self.dtype)


class VehicleState:
    dtype = [
        ('time', time_type),  # timestamp
        ('q', float_type, 4),  # quaternion
        ('b', float_type, 3),  # gyro bias
        ('omega', float_type, 3),  # angular velocity
        ('pos', float_type, 3),  # position
        ('vel', float_type, 3),  # velocity
        ('accel', float_type, 3),  # acceleration
    ]

    def __init__(self):
        self.data = init_data(self.dtype)


class EstimatorStatus():
    n_max = 20
    dtype = [
        ('time', time_type),  # timestamp
        ('elapsed', float_type),  # elapsed time
        ('n_x', 'i4'),  # number of states
        ('x', float_type, n_max),  # states array
        ('W', float_type, n_max),  # W matrix diagonal (sqrt(P))
        ('r_mag', float_type),  # magnetometer residual
        ('r_std_mag', float_type),  # magnetometer residual standard deviation
        ('beta_mag', float_type),  # magnetometer fault detection
        ('mag_ret', 'i8'),  # mag return code
        ('r_accel', float_type),  # accelerometer residual
        ('r_std_accel', float_type),  # accelerometer residual standard deviation
        ('beta_accel', float_type),  # accelerometer fault detection
        ('accel_ret', 'i8'),  # accelerometer return code
    ]

    def __init__(self):
        self.data = init_data(self.dtype)


class Params():

    def __init__(self, core):
        self.dtype = [('time', time_type)]
        for name, p in core._declared_params.items():
            self.dtype.append((p.name, p.dtype))
        self.data = init_data(self.dtype)
        for name, p in core._declared_params.items():
            self.data[name] = p.value


class Log():

    def __init__(self, core):
        self.dtype = [('time', time_type)]
        for topic, publisher in core._publishers.items():
            if not hasattr(publisher.msg_type, 'dtype'):
                msg = publisher.msg_type(core)
                self.dtype.append((topic, msg.dtype))
            else:
                self.dtype.append((topic, publisher.msg_type.dtype))
        self.data = init_data(self.dtype)


