from pyecca.so3.mrp import Mrp
import casadi as ca

tol = 1e-5  # tolerance

a = Mrp([0.1, 0.2, 0.3])
b = Mrp([0.2, 0.1, 0.4])


def test_dcm():
    assert ca.norm_fro(Mrp.from_dcm(Mrp.to_dcm(a)) - a) < tol


def test_exp_log():
    assert ca.norm_fro(Mrp.log(Mrp.exp(a)) - a) < tol


def test_shadow():
    # TODO
    a.shadow()


# noinspection PyPep8Naming
def test_product():
    assert ca.norm_fro(a * -a) < tol
    Ra = a.to_dcm()
    assert ca.norm_fro(a - Mrp.from_dcm(Ra)) < tol
    Rb = b.to_dcm()
    assert ca.norm_fro(b - Mrp.from_dcm(Rb)) < tol
    assert ca.norm_fro(ca.mtimes(Ra, Rb) - (a * b).to_dcm()) < tol
    print(type(1 * b))


def test_inv():
    assert ca.norm_fro(a.inv() + a) < tol


def test_derivative():
    assert ca.norm_fro(a.derivative(ca.SX([0, 0, 0])) - [0, 0, 0]) < 1e-5
