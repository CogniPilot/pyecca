from pyecca.dcm import Dcm
import casadi as ca
import pytest

tol = 1e-5 # tolerance


a = Dcm([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
b = Dcm([[0, 1, 0], [1, 0, 0], [0, 0, 1]])


def test_product():
    c = a*b
    assert True


def test():
    with pytest.raises(AssertionError):
        a + b
    with pytest.raises(AssertionError):
        a - b
    with pytest.raises(AssertionError):
        c = -a
