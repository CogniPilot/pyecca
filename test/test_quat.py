from pyecca.so3.dcm import Dcm
from pyecca.so3.quat import Quat
import casadi as ca
import pytest

tol = 1e-5  # tolerance

# Analytical Mechanics of Space Systems, Shaub pg. 97
dcm_check = Dcm([
    [0.892539, 0.157379, -0.422618],
    [-0.275451, 0.932257, -0.234570],
    [0.357073, 0.325773, 0.875426]
]).T  # transpose from Schaub dcm convention
# direction of transform is reversed
q_check = Quat([0.961798, -0.14565, 0.202665, 0.112505])
a = Quat([1, 0, 0, 0])
b = Quat([0, 1, 0, 0])


def test_to_from_dcm():
    assert ca.norm_fro(Quat.from_dcm(dcm_check) - q_check) < tol
    assert ca.norm_fro(Quat.to_dcm(q_check) - dcm_check) < tol


def test_product():
    print(type(a * b))
    assert ca.norm_fro((a * b).to_dcm() - ca.mtimes(a.to_dcm(), b.to_dcm())) < tol


def test_ctor():
    Quat()
    with pytest.raises(AssertionError):
        Quat([1, 2, 3])


def test_arithmetic():
    assert ca.norm_fro(a + b - Quat([1, 1, 0, 0])) < tol
    assert ca.norm_fro(a - b - Quat([1, -1, 0, 0])) < tol
