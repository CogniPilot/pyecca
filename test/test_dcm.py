from pyecca.so3.dcm import Dcm
import pytest

tol = 1e-5  # tolerance

a = Dcm([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
b = Dcm([[0, 1, 0], [1, 0, 0], [0, 0, 1]])


def test_product():
    c = a * b
    assert c.shape == (3, 3)
    assert True


def test():
    with pytest.raises(AssertionError):
        a + b
    with pytest.raises(AssertionError):
        a - b
    with pytest.raises(AssertionError):
        # noinspection PyUnusedLocal
        q = -a
