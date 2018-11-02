import casadi as ca
from pyecca2.lie import so3, r3
from pyecca2.lie.so3 import quat, dcm, mrp, euler

eps = 1e-10


def test_so3():

    r = ca.DM([0.1, 0.2, 0.3, 0])
    v = ca.DM([0.1, 0.2, 0.3])
    q1 = ca.DM([1, 0, 0, 0])
    R = ca.SX.eye(3)

    assert ca.norm_2(dcm.log(dcm.exp(v)) - v) < eps
    assert ca.norm_2(quat.log(quat.exp(v)) - v) < eps
    assert ca.norm_2(mrp.log(mrp.exp(v)) - v) < eps

    assert ca.norm_2(so3.vee(so3.wedge(v)) - v) < eps

    assert ca.norm_2(quat.from_dcm(dcm.from_quat(q1)) - q1) < eps
    assert ca.norm_2(quat.from_mrp(mrp.from_quat(q1)) - q1) < eps
    assert ca.norm_2(quat.from_euler(euler.from_quat(q1)) - q1) < eps

    assert ca.norm_2(mrp.from_dcm(dcm.from_mrp(r)) - r) < eps
    assert ca.norm_2(mrp.from_quat(quat.from_mrp(r)) - r) < eps
    assert ca.norm_2(mrp.from_euler(euler.from_mrp(r)) - r) < eps

    assert ca.norm_fro(dcm.from_quat(quat.from_dcm(R)) - R) < eps
    assert ca.norm_fro(dcm.from_mrp(mrp.from_dcm(R)) - R) < eps
    assert ca.norm_fro(dcm.from_euler(euler.from_dcm(R)) - R) < eps


def test_r3():
    v = ca.DM([0.1, 0.2, 0.3])
    print(r3.product(v, r3.inv(v)))

    def f(m1, m2):
        v = ca.SX([1, 2, 3])

        print(m1.exp(v), m2.exp(v))

    f(so3.quat, r3)
