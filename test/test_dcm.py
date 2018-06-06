from pyecca.so3.dcm import Dcm
import pytest
import casadi as ca


tol = 1e-5  # tolerance

a = Dcm([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
b = Dcm([[0, 1, 0], [1, 0, 0], [0, 0, 1]])


def test_product():
    c = a * b
    assert c.shape == (3, 3)
    assert True


def test_log_exp():
    omega = [0.1, 0.2, 0.3]
    assert ca.norm_fro(Dcm.exp(omega).log()  - omega) < tol


def test_derivative():
    assert ca.norm_fro(a.derivative(ca.SX([0, 0, 0])) - [0, 0, 0]) < 1e-5