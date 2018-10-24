import numpy as np


def init_data(dtype):
    return np.zeros(1, dtype=dtype)[0]


class Imu:
    dtype = [
        ('time', 'i4'),  # timestamp
        ('gyro', 'f4', 3),  # gyroscope measurement
        ('mag', 'f4', 3),  # magnetometer measurement
        ('accel', 'f4', 3),  # accelerometer measurement
    ]

    def __init__(self):
        self.data = init_data(self.dtype)


class SimState:
    dtype = [
        ('time', 'i4'),  # timestamp
        ('q', 'f4', 4),  # quaternion
        ('omega', 'f4', 3),  # angular velocity
        ('pos', 'f4', 3),  # position
        ('vel', 'f4', 3),  # velocity
        ('accel', 'f4', 3),  # acceleration
    ]

    def __init__(self):
        self.data = init_data(self.dtype)


class EstimatorStatus():
    n_max = 20
    nP_max = int(20 * (20 - 1) / 2)
    dtype = [
        ('time', 'f4'),  # timestamp
        ('n_x', 'i4'),  # number of states
        ('x', 'f4', n_max),  # states array
        ('P', 'f4', nP_max),  # P matrix upper triangle
        ('beta_mag', 'f4'),  # magnetometer fault detection
        ('beta_accel', 'f4'),  # accelerometer fault detection
        ('beta_gyro', 'f4'),  # gyroscope fault detection
    ]

    def __init__(self):
        self.data = init_data(self.dtype)


class Params():
    dtype = [
        ('time', 'f4'),  # timestamp
        ('logger/dt', 'f4')
    ]

    def __init__(self):
        self.data = init_data(self.dtype)


class Log():

    def __init__(self, core):
        self.dtype = [('time', 'f4')]
        for topic, publisher in core._publishers.items():
            self.dtype.append((topic, publisher.msg_type.dtype))
        self.data = init_data(self.dtype)


