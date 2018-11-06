import numpy as np

# avoid integer types, they don't work well with plotting,
# no need to be overly cautious with memory here


time_type = 'f8'
float_type = 'f8'  # change this to f4 to simulate with 32 bit precision


def init_data(dtype):
    """
    Initializes all data to nan
    :param dtype: the numpy dtype
    :return: the initialized data
    """
    data = np.zeros(1, dtype=dtype)[0]
    data.fill(np.nan)
    return data


class Msg:

    def __init__(self, dtype: np.dtype):
        self.data = np.zeros(1, dtype=dtype)[0]
        self.data.fill(np.nan)

    def __repr__(self):
        return repr(self.data)


class Imu(Msg):
    dtype = np.dtype([
        ('time', time_type),  # timestamp
        ('gyro', float_type, 3),  # gyroscope measurement
        ('accel', float_type, 3),  # accelerometer measurement
    ])

    def __init__(self):
        super().__init__(self.dtype)


class Mag(Msg):
    dtype = np.dtype([
        ('time', time_type),  # timestamp
        ('mag', float_type, 3),  # magnetometer measurement
    ])

    def __init__(self):
        super().__init__(self.dtype)


class VehicleState(Msg):
    dtype = np.dtype([
        ('time', time_type),  # timestamp
        ('q', float_type, 4),  # quaternion
        ('r', float_type, 4),  # mrp
        ('b', float_type, 3),  # gyro bias
        ('omega', float_type, 3),  # angular velocity
        ('pos', float_type, 3),  # position
        ('vel', float_type, 3),  # velocity
        ('accel', float_type, 3),  # acceleration
    ])

    def __init__(self):
        super().__init__(self.dtype)


class EstimatorStatus(Msg):
    n_max = 20
    dtype = np.dtype([
        ('time', time_type),  # timestamp
        ('cpu_predict', time_type),  # elapsed cpu prediction time
        ('cpu_mag', time_type),  # elapsed cpu mag correction time
        ('cpu_accel', time_type),  # elapsed cpu accel correction time
        ('n_x', float_type),  # number of states
        ('x', float_type, n_max),  # states array
        ('W', float_type, n_max),  # W matrix diagonal (sqrt(P))
        ('r_mag', float_type, 3),  # magnetometer residual
        ('r_std_mag', float_type, 3),  # magnetometer residual standard deviation
        ('beta_mag', float_type),  # magnetometer fault detection
        ('mag_ret', float_type),  # mag return code
        ('r_accel', float_type, 3),  # accelerometer residual
        ('r_std_accel', float_type, 3),  # accelerometer residual standard deviation
        ('beta_accel', float_type),  # accelerometer fault detection
        ('accel_ret', float_type),  # accelerometer return code
    ])

    def __init__(self):
        super().__init__(self.dtype)


class Params(Msg):

    def __init__(self, core):
        dtype = [('time', time_type)]
        for name, p in core._declared_params.items():
            dtype.append((p.name, p.dtype))
        self.dtype = np.dtype(dtype)
        super().__init__(self.dtype)

        for name, p in core._declared_params.items():
            self.data[name] = p.value


class Log(Msg):

    def __init__(self, core):
        dtype = [('time', time_type)]
        for topic, publisher in core._publishers.items():
            if not hasattr(publisher.msg_type, 'dtype'):
                msg = publisher.msg_type(core)
                dtype.append((topic, msg.dtype))
            else:
                dtype.append((topic, publisher.msg_type.dtype))
        self.dtype = np.dtype(dtype)
        super().__init__(self.dtype)
