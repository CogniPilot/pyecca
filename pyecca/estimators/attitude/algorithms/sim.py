from .common import *

# x, state (7)
# -----------
# r, mrp (3)
# s, shadow, mrp shadow state (1)
# b, gyro bias (3)
x = ca.SX.sym("x", 7)
r = x[0:4]  # last state is shadow state
b_gyro = x[4:7]
q = so3.Quat.from_mrp(r)
C_nb = so3.Dcm.from_mrp(r)


def get_state(**kwargs):
    return ca.Function("get_state", [x], [q, r, b_gyro], ["x"], ["q", "r", "b_gyro"])


def simulate(**kwargs):
    # state derivative
    xdot = ca.vertcat(so3.Mrp.kinematics(r, omega_t), std_gyro_rw * w_gyro_rw)
    f_xdot = ca.Function(
        "xdot",
        [t, x, omega_t, sn_gyro_rw, w_gyro_rw],
        [xdot],
        ["t", "x", "omega_t", "sn_gyro_rw", "w_gyro_rw"],
        ["xdot"],
    )

    # state prop with noise
    x1_sim = util.rk4(
        lambda t, x: f_xdot(t, x, omega_t, sn_gyro_rw, w_gyro_rw), t, x, dt
    )
    x1_sim[:4] = so3.Mrp.shadow_if_necessary(x1_sim[:4])
    return ca.Function(
        "simulate",
        [t, x, omega_t, sn_gyro_rw, w_gyro_rw, dt],
        [x1_sim],
        ["t", "x", "omega_t", "sn_gyro_rw", "w_gyro_rw", "dt"],
        ["x1"],
    )


def measure_gyro(**kwargs):
    return ca.Function(
        "measure_gyro",
        [x, omega_t, std_gyro, w_gyro],
        [omega_t + b_gyro + w_gyro * std_gyro],
        ["x", "omega_t", "std_gyro", "w_gyro"],
        ["y"],
    )


def measure_mag(**kwargs):
    C_nm = so3.Dcm.product(so3.Dcm.exp(mag_decl * e3), so3.Dcm.exp(-mag_incl * e2))
    B_n = mag_str * ca.mtimes(C_nm, ca.SX([1, 0, 0]))
    return ca.Function(
        "measure_mag",
        [x, mag_str, mag_decl, mag_incl, std_mag, w_mag],
        [ca.mtimes(C_nb.T, B_n) + w_mag * std_mag],
        ["x", "mag_str", "mag_decl", "mag_incl", "std_mag", "w_mag"],
        ["y"],
    )


def measure_accel(**kwargs):
    return ca.Function(
        "measure_accel",
        [x, g, std_accel, w_accel],
        [g * ca.mtimes(C_nb.T, ca.SX([0, 0, -1])) + w_accel * std_accel],
        ["x", "g", "std_accel", "w_accel"],
        ["y"],
    )


def constants(**kwargs):
    x0 = ca.DM([0.1, 0.2, 0.3, 0, 0, 0, 0.01])
    return ca.Function("constants", [], [x0], [], ["x0"])


def rotation_error(**kwargs):
    q1 = ca.SX.sym("q1", 4)
    q2 = ca.SX.sym("q2", 4)
    dq = so3.Quat.product(so3.Quat.inv(q1), q2)
    xi = so3.Quat.log(dq)
    return ca.Function("rotation_error", [q1, q2], [xi], ["q1", "q2"], ["xi"])


def eqs(**kwargs):
    return {
        "simulate": simulate(**kwargs),
        "measure_gyro": measure_gyro(**kwargs),
        "measure_accel": measure_accel(**kwargs),
        "measure_mag": measure_mag(**kwargs),
        "constants": constants(**kwargs),
        "get_state": get_state(**kwargs),
        "rotation_error": rotation_error(**kwargs),
    }
