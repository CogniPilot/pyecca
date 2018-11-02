import casadi as ca
from pyecca2.lie import so3, r3
from pyecca2.lie.util import DirectProduct
from pyecca2.lie.so3 import Quat, Dcm, Euler, Mrp
from pyecca2.lie.r3 import R3

eps = 1e-10


def test_so3():

    r = ca.DM([0.1, 0.2, 0.3, 0])
    v = ca.DM([0.1, 0.2, 0.3])
    q1 = ca.DM([1, 0, 0, 0])
    R = ca.SX.eye(3)

    assert ca.norm_2(Dcm.log(Dcm.exp(v)) - v) < eps
    assert ca.norm_2(Quat.log(Quat.exp(v)) - v) < eps
    assert ca.norm_2(Mrp.log(Mrp.exp(v)) - v) < eps

    assert ca.norm_2(so3.vee(so3.wedge(v)) - v) < eps

    assert ca.norm_2(Quat.from_dcm(Dcm.from_quat(q1)) - q1) < eps
    assert ca.norm_2(Quat.from_mrp(Mrp.from_quat(q1)) - q1) < eps
    assert ca.norm_2(Quat.from_euler(Euler.from_quat(q1)) - q1) < eps

    assert ca.norm_2(Mrp.from_dcm(Dcm.from_mrp(r)) - r) < eps
    assert ca.norm_2(Mrp.from_quat(Quat.from_mrp(r)) - r) < eps
    assert ca.norm_2(Mrp.from_euler(Euler.from_mrp(r)) - r) < eps

    assert ca.norm_fro(Dcm.from_quat(Quat.from_dcm(R)) - R) < eps
    assert ca.norm_fro(Dcm.from_mrp(Mrp.from_dcm(R)) - R) < eps
    assert ca.norm_fro(Dcm.from_euler(Euler.from_dcm(R)) - R) < eps


def test_direct_product():
    MrpxR3xR3xR3 = DirectProduct([Mrp, R3, R3, R3])
    g1 = ca.SX.sym('g1', *MrpxR3xR3xR3.group_shape)
    g2 = ca.SX.sym('g1', *MrpxR3xR3xR3.group_shape)
    print(MrpxR3xR3xR3.product(g1, g2))


def test_r3():
    v1 = ca.SX([1, 2, 3])
    v2 = ca.SX([4, 5, 6])
    v3 = v1 + v2
    assert ca.norm_2(R3.product(v1, v2) - v3) < eps