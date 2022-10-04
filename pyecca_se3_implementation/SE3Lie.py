import casadi as ca
from pyecca.lie import so3, r3, se3
from pyecca.lie.util import DirectProduct
from pyecca.lie.so3 import Quat, Dcm, Euler, Mrp
from pyecca.lie.r3 import R3
from pyecca.lie.se3 import SE3
from pyecca.test import test_lie
import numpy as np

eps = 1e-7 # to avoid divide by zero
'''
SE3Lie extension with applications of casadi
'''

class se3: #se3 Lie Algebra Functions
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
    def ad_matrix(cls, v): #input vee operator [x,y,z,theta1,theta2,theta3]
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

    @classmethod
    def check_group_shape(cls, a):
        assert a.shape == cls.group_shape or a.shape == (cls.group_shape[0],)

    @classmethod
    def vee(cls, X):
        '''
        This takes in an element of the SE3 Lie Group (Wedge Form) and returns the se3 Lie Algebra elements 
        '''
        v = ca.SX(6, 1) #in the form [theta1, theta2, theta3, x, y, z] ##NEEDS CORRECTION into form [x,y,z,theta1,theta2,theta3]???
        v[0, 0] = X[2, 1] #review if any of these switches will be affected in se3.py and so3.py
        v[1, 0] = X[0, 2]
        v[2, 0] = X[1, 0]
        v[3, 0] = X[0, 3]
        v[4, 0] = X[1, 3]
        v[5, 0] = X[2, 3]
        return v

    @classmethod
    def wedge(cls, v):
        '''
        This takes in an element of the se3 Lie Algebra and returns the se3 Lie Algebra matrix
        '''
        X = ca.SX.zeros(4, 4) #in the form [theta1, theta2, theta3, x, y, z] ##NEEDS CORRECTION into form [x,y,z,theta1,theta2,theta3]???
        X[0, 1] = -v[2]
        X[0, 2] = v[1]
        X[1, 0] = v[2]
        X[1, 2] = -v[0]
        X[2, 0] = -v[1]
        X[2, 1] = v[0]
        X[0, 3] = v[3]
        X[1, 3] = v[4]
        X[2, 3] = v[5]
        return X        

    @classmethod
    def matmul(cls, a, b):
        cls.check_group_shape(a)
        cls.check_group_shape(b)
        return ca.mtimes(a, b)

    @classmethod
    def exp(cls, v): #accept input in wedge operator form
        v = cls.vee(v)
        v_so3 = v[:3] #grab only rotation terms for so3 uses ##WILL NEED TO BE CHANGED into v_so3 = v[3:6]
        X_so3 = so3.wedge(v_so3) #wedge operator for so3
        theta = ca.norm_2(so3.vee(X_so3)) #theta term using norm for sqrt(theta1**2+theta2**2+theta3**2)
        
        # translational components u
        u = ca.SX(3, 1)
        u[0, 0] = v[3] #change to v[0]
        u[1, 0] = v[4] #change to v[1]
        u[2, 0] = v[5] #change to v[2]
        
        R = so3.Dcm.exp(v_so3) #'Dcm' for direction cosine matrix representation of so3 LieGroup Rotational
        V = ca.SX.eye(3) + cls.C2(theta)*X_so3 + cls.C4(theta)*ca.mtimes(X_so3, X_so3)
        horz = ca.horzcat(R, ca.mtimes(V,u))
        
        lastRow = ca.horzcat(0,0,0,1)
        
        return ca.vertcat(horz, lastRow)

class SE3:
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
    
    @classmethod
    def one(cls):
        return ca.SX.zeros(6, 1)
    
    @classmethod
    def product(cls, a, b):
        cls.check_group_shape(a)
        cls.check_group_shape(b)
        return ca.mtimes(a, b)

    @classmethod
    def inv(cls, a): #input a matrix of SX form from casadi
        cls.check_group_shape(a)
        a_inv = ca.solve(a,ca.SX.eye(6)) #Google Group post mentioned ca.inv() could take too long, and should explore solve function
        return ca.transpose(a)
    
    @classmethod
    def log(cls, G):
        theta = ca.arccos(((G[0, 0]+G[1, 1]+G[2, 2]) - 1) / 2)
        wSkew = so3.wedge(so3.Dcm.log(G[:3,:3]))
        V_inv = ca.SX.eye(3) - 0.5*wSkew + (1/(theta**2))*(1-((cls.C1(theta))/(2*cls.C2(theta))))*ca.mtimes(wSkew, wSkew)
        
        # t is translational component vector
        t = ca.SX(3, 1)     
        t[0, 0] = G[0, 3]
        t[1, 0] = G[1, 3]
        t[2, 0] = G[2, 3]
        
        uInv = ca.mtimes(V_inv, t) 
        horz2 = ca.horzcat(wSkew, uInv)
        lastRow2 = ca.horzcat(0,0,0,0)
        return ca.vertcat(horz2, lastRow2)

# def se3_diff_correction(v): #U Matrix for se3 with input vee operator
# --- could utilize series form by implementing C1 and C2 ---
#       #To do


def se3_diff_correction_inv(v): #U_inv of se3 input vee operator
    x = ca.SX.sym('x')
    C1 = ca.Function('a', [x], [ca.if_else(ca.fabs(x) < eps, 1 - x ** 2 / 6 + x ** 4 / 120, ca.sin(x)/x)])
    C2 = ca.Function('b', [x], [ca.if_else(ca.fabs(x) < eps, 0.5 - x ** 2 / 24 + x ** 4 / 720, (1 - ca.cos(x)) / x ** 2)])
    del x
    # v = se3.vee(v)  #This only applies if v is inputed from Lie Group format
    v_so3 = v[3:6] #grab only rotation terms for so3 uses ### might need to be changed to v[3:6]
    X_so3 = so3.wedge(v_so3) #wedge operator for so3
    theta = ca.norm_2(so3.vee(X_so3)) #theta term using norm for sqrt(theta1**2+theta2**2+theta3**2)

    u_inv = ca.SX(6, 6)
    u_inv[0,0] = C2(theta)*(-v[4,0]**2 - v[5,0]**2) + 1
    u_inv[0,1] = -C1(theta)*v[5,0] + C2(theta)*v[3,0]*v[4,0]
    u_inv[0,2] = C1(theta) * v[4,0] + C2(theta) * v[3,0]*v[4,0]
    u_inv[0,3] = C2(theta) * (-2*v[4,0]*v[1,0]-2*v[5,0]*v[2,0])
    u_inv[0,4] = -C1(theta) * v[2,0] + C2(theta)*(v[4,0]*v[0,0]+v[5,0]*v[1,0])
    u_inv[0,5] = C1(theta) * v[1,0] + C2(theta)*(v[3,0]*v[2,0]+v[5,0]*v[0,0])
    
    u_inv[1,0] = C1(theta) * v[5,0] + C2(theta) * v[3,0] * v[4,0]
    u_inv[1,1] = C2(theta) *(-v[3,0]**2 - v[5,0]**2)+1
    u_inv[1,2] = -C1(theta)*v[3,0] + C2(theta) * v[4,0]*v[5,0]
    u_inv[1,3] = C1(theta) * v[2,0] + C2(theta) * (v[3,0]*v[1,0]+v[4,0]*v[0,0])
    u_inv[1,4] = C2(theta)* (-v[3,0] * v[0,0] - v[5,0]*v[0,0]-2*v[5,0]*v[2,0]) ##Check syntax
    u_inv[1,5] = -C1(theta) * v[0,0] + C2(theta) * (v[4,0]*v[2,0] + v[5,0] *v[1,0])

    u_inv[2,0] = -C1(theta) * v[4,0] + C2(theta) * v[3,0] * v[5,0]
    u_inv[2,1] = C1(theta) * v[3,0] + C2(theta) * v[4,0] * v[5,0]
    u_inv[2,2] = C2(theta) * (-v[3,0] **2  - v[4,0]**2) +1
    u_inv[2,3] = -C1(theta) * v[1,0] + C2(theta) * (v[3,0]*v[2,0] + v[5,0]*v[0,0])
    u_inv[2,4] = C1(theta) * v[0,0] + C2(theta) * (v[4,0]*v[2,0] + v[5,0] *v[1,0])
    u_inv[2,5] = C2(theta) * (-2*v[3,0]*v[0,0] - 2*v[4,0] *v[1,0])

    u_inv[3,3] = C2(theta) * (- v[4,0]**2 - v[5,0]**2) +1
    u_inv[3,4] = -C1(theta)*v[5,0] + C2(theta)*v[4,0]*v[5,0]
    u_inv[3,5] = C1(theta) * v[4,0] + C2(theta) * v[3,0] * v[5,0]

    u_inv[4,3] = C1(theta) * v[5,0] + C2(theta) * v[3,0] * v[4,0]
    u_inv[4,4] = C2(theta) * (-v[3,0]*v[5,0] - v[5,0]**2) +1
    u_inv[4,5] = -C1(theta) * v[3,0] + C2(theta) * v[4,0] *v[5,0]

    u_inv[5,3] = -C1(theta) * v[4,0] + C2(theta) * v[5,0]**2
    u_inv[5,4] = C1(theta) * v[5,0] + C2(theta) * v[4,0] * v[5,0]
    u_inv[5,5] = C2(theta) * (-v[3,0] * v[5,0] - v[4,0]**2)+1
    return u_inv