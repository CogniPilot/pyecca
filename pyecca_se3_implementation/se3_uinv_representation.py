import casadi as ca
from pyecca.lie import so3, r3, se3
from pyecca.lie.util import DirectProduct
from pyecca.lie.so3 import Quat, Dcm, Euler, Mrp
from pyecca.lie.r3 import R3
from pyecca.lie.se3 import SE3
from pyecca.test import test_lie
import numpy as np


'''
This code intends to perform U_inv matrix of se3
'''

class SE3_diff_corr:
  group_shape = (6, 6)
  group_params = 12
  algebra_params = 6

# coefficients 
  x = ca.SX.sym('x')
  C1 = ca.Function('a', [x], [ca.if_else(ca.fabs(x) < eps, 1 - x ** 2 / 6 + x ** 4 / 120, ca.sin(x)/x)])
  C2 = ca.Function('b', [x], [ca.if_else(ca.fabs(x) < eps, 0.5 - x ** 2 / 24 + x ** 4 / 720, (1 - ca.cos(x)) / x ** 2)])
  del x

  def __init__(self, SO3=None):
    if SO3 == None:
      self.SO3 = so3.Dcm()

  @classmethod
  def check_group_shape(cls, a):
    assert a.shape == cls.group_shape or a.shape == (cls.group_shape[0],)

  @classmethod #takes 6x1 lie algebra
  def adjoint_se3(cls, v): #input vee operator
    ad_se3 = ca.SX(6, 6)
    ad_se3[0,1] = -v[5,0]
    ad_se3[0,2] = v[3,0]
    ad_se3[0,4] = -v[2,0]
    ad_se3[0,5] = v[1,0]
    ad_se3[1,0] = v[5,0]
    ad_se3[1,2] = -v[3,0]
    ad_se3[1,3] = v[2,0]
    ad_se3[1,5] = -v[0,0]  
    ad_se3[2,0] = -v[4,0]
    ad_se3[2,1] = v[3,0]
    ad_se3[2,3] = -v[1,0]
    ad_se3[2,4] = v[0,0]
    ad_se3[3,4] = -v[5,0]
    ad_se3[3,5] = v[4,0]
    ad_se3[4,3] = v[5,0]
    ad_se3[4,5] = -v[3,0]
    ad_se3[5,3] = -v[4,0]
    ad_se3[5,4] = v[3,0]
    return ad_se3
    
    

  @classmethod #Does these inputs using vee operator work? Seems like there should be a better way
  def se3_diff_correction_inv(cls, v, C1,C2): #input vee operator
    u_inv = ca.SX(6, 6)
    #u_inv[0,0] = C2*(v[3,0]**2 - v[])
    
    #is it possible to combine theta1, theta2, theta3 into theta?

