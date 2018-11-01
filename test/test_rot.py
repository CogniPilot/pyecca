import casadi as ca
from pyecca2 import so3
from pyecca2.so3 import quat, dcm, mrp, euler


def test():

    v = ca.DM([0.1, 0.2, 0.3])
    q1 = ca.DM([1, 0, 0, 0])
    R = ca.SX.eye(3)

    eps = 1e-10

    assert ca.norm_2(dcm.log(dcm.exp(v)) - v) < eps
    assert ca.norm_2(quat.log(quat.exp(v)) - v) < eps
    assert ca.norm_2(mrp.log(mrp.exp(v)) - v) < eps

    assert ca.norm_2(so3.vee(so3.wedge(v)) - v) < eps

    assert ca.norm_2(quat.from_dcm(dcm.from_quat(q1)) - q1) < eps
    assert ca.norm_2(quat.from_mrp(mrp.from_quat(q1)) - q1) < eps
    assert ca.norm_2(quat.from_euler(euler.from_quat(q1)) - q1) < eps

    assert ca.norm_2(mrp.from_dcm(dcm.from_mrp(v)) - v) < eps
    assert ca.norm_2(mrp.from_quat(quat.from_mrp(v)) - v) < eps
    assert ca.norm_2(mrp.from_euler(euler.from_mrp(v)) - v) < eps

    assert ca.norm_fro(dcm.from_quat(quat.from_dcm(R)) - R) < eps
    assert ca.norm_fro(dcm.from_mrp(mrp.from_dcm(R)) - R) < eps
    assert ca.norm_fro(dcm.from_euler(euler.from_dcm(R)) - R) < eps

