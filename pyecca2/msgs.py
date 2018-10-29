import numpy as np


def init_data(dtype):
    return np.zeros(1, dtype=dtype)[0]


class Imu:
    dtype = [
        ('time', 'f4'),  # timestamp
        ('gyro', 'f4', 3),  # gyroscope measurement
        ('accel', 'f4', 3),  # accelerometer measurement
    ]

    def __init__(self):
        self.data = init_data(self.dtype)


class Mag:
    dtype = [
        ('time', 'f4'),  # timestamp
        ('mag', 'f4', 3),  # magnetometer measurement
    ]

    def __init__(self):
        self.data = init_data(self.dtype)


class VehicleState:
    dtype = [
        ('time', 'i4'),  # timestamp
        ('q', 'f4', 4),  # quaternion
        ('b', 'f4', 3),  # gyro bias
        ('omega', 'f4', 3),  # angular velocity
        ('pos', 'f4', 3),  # position
        ('vel', 'f4', 3),  # velocity
        ('accel', 'f4', 3),  # acceleration
    ]

    def __init__(self):
        self.data = init_data(self.dtype)


class EstimatorStatus():
    n_max = 20
    dtype = [
        ('time', 'f4'),  # timestamp
        ('elapsed', 'f4'),  # elapsed time
        ('n_x', 'i4'),  # number of states
        ('x', 'f4', n_max),  # states array
        ('W', 'f4', n_max),  # W matrix diagonal (sqrt(P))
        ('beta_mag', 'f4'),  # magnetometer fault detection
        ('beta_accel', 'f4'),  # accelerometer fault detection
        ('beta_gyro', 'f4'),  # gyroscope fault detection
    ]

    def __init__(self):
        self.data = init_data(self.dtype)


class Params():

    def __init__(self, core):
        self.dtype = [('time', 'f4')]
        for name, p in core._declared_params.items():
            self.dtype.append((p.name, p.dtype))
        self.data = init_data(self.dtype)
        for name, p in core._declared_params.items():
            self.data[name] = p.value


class Log():

    def __init__(self, core):
        self.dtype = [('time', 'f4')]
        for topic, publisher in core._publishers.items():
            if not hasattr(publisher.msg_type, 'dtype'):
                msg = publisher.msg_type(core)
                self.dtype.append((topic, msg.dtype))
            else:
                self.dtype.append((topic, publisher.msg_type.dtype))
        self.data = init_data(self.dtype)


