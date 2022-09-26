import casadi as ca
from pyecca.lie import so3, r3, se3
from pyecca.lie.util import DirectProduct
from pyecca.lie.so3 import Quat, Dcm, Euler, Mrp
from pyecca.lie.r3 import R3
from pyecca.lie.se3 import SE3
from pyecca.test import test_lie
import numpy as np


'''
This initialize the matrix representation for se3 lie algebra elements
'''
  #erms = ca.SX([0.1, 0.2, 0.3, 4, 5, 6]) # terms : [theta1, theta2, theta3, x,y,z]
v = ca.DM([0.1, 0.2, 0.3]) #translational component
R = Dcm.from_euler(ca.DM([0.1, 0.2, 0.3])) #Rotational component using Dcm

R = np.array(ca.DM(R))
print(f"{v} and R is {R}")
# lie_elements = ca.horzcat(np.transpose(v),R)


'''
lie_alg_se3= ca.SX(4, 4)
lie_alg_se3[0, 1] = -R[2]
lie_alg_se3[0, 2] = R[1]
lie_alg_se3[1, 0] = R[2]
lie_alg_se3[1, 2] = R[0]
lie_alg_se3[2, 0] = -R[1]
lie_alg_se3[2, 1] = R[0]
lie_alg_se3[0, 4] = v[0]
lie_alg_se3[1, 4] = v[1]
lie_alg_se3[2, 4] = v[2]
'''
