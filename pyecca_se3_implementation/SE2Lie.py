import casadi as ca
from pyecca.lie import so3, r3, se3
from pyecca.lie.util import DirectProduct
from pyecca.lie.so3 import Quat, Dcm, Euler, Mrp
from pyecca.lie.r3 import R3
from pyecca.lie.se3 import SE3
from pyecca.test import test_lie
import numpy as np
from casadi.tools.graph import graph, dotdraw
import os
import pydot


eps = 1e-7 # to avoid divide by zero
'''
SE2Lie with applications of casadi
'''

class se2: #se2 Lie Algebra Function
    group_shape = (3, 3)
    group_params = 6
    algebra_params = 3

    def __init__(self, SO2=None):
        pass

    @classmethod
    def check_group_shape(cls, a):
        assert a.shape == cls.group_shape or a.shape == (cls.group_shape[0],)

    @classmethod #takes 6x1 lie algebra
    def ad_matrix(cls, v): #input vee operator [x,y,theta]
        ad_se2 = ca.SX(3, 3)
        ad_se2[0,1] = -v[2]
        ad_se2[0,2] = v[1]
        ad_se2[1,0] = v[2]
        ad_se2[1,2] = -v[0] 
        return ad_se2

    @classmethod
    def check_group_shape(cls, a):
        assert a.shape == cls.group_shape or a.shape == (cls.group_shape[0],)

    @classmethod
    def vee(cls, X):
        '''
        This takes in an element of the SE2 Lie Group (Wedge Form) and returns the se2 Lie Algebra elements 
        '''
        v = ca.SX(3, 1) #[x,y,theta]
        v[0,0] = -X[1, 2]
        v[1,0] = X[0, 2]
        v[2,0] = X[1, 0]
        return v

    @classmethod
    def wedge(cls, v): #input v = [x,y,theta]
        '''
        This takes in an element of the se2 Lie Algebra and returns the se2 Lie Algebra matrix
        '''
        X = ca.SX.zeros(3, 3)
        X[0, 1] = -v[2]
        X[1, 0] = v[2]
        X[0, 2] = v[0]
        X[1, 2] = v[1]
        return X        

    @classmethod
    def matmul(cls, a, b):
        cls.check_group_shape(a)
        cls.check_group_shape(b)
        return ca.mtimes(a, b)

    @classmethod
    def exp(cls, v): #accept input in wedge operator form
        v = cls.vee(v)
        theta = v[2]

        # translational components u
        u = ca.SX(2, 1)
        u[0, 0] = v[0]
        u[1, 0] = v[1]

        if type(v[1]) == 'int' and theta < eps:
            a = 1 - theta ** 2 / 6 + theta ** 4 / 120
            b = 0.5 - theta ** 2 / 24 + theta ** 4 / 720
        else:
            a = ca.sin(theta)/theta
            b = (1-ca.cos(theta)/theta)

        V = ca.SX(2,2)
        V[0,0] = a
        V[0,1] = -b
        V[1,0] = b
        V[1,1] = a

        if type(v[1]) == 'int' and theta < eps:
            a = theta - theta **3 / 6 + theta ** 5 / 120
            b = 1 - theta**2 / 2 + theta ** 4 / 24
        else:
            a = ca.sin(theta)
            b = ca.cos(theta)

        R = ca.SX(2,2) #Exp(wedge(theta))
        R[0,0] = b
        R[0,1] = -a
        R[1,0] = a
        R[1,1] = b

        horz = ca.horzcat(R, ca.mtimes(V,u))
        
        lastRow = ca.horzcat(0,0,1)
        
        return ca.vertcat(horz, lastRow)


class SE2:
    group_shape = (3, 3)
    group_params = 6
    algebra_params = 3

    def __init__(self):
        pass

    @classmethod
    def check_group_shape(cls, a):
        assert a.shape == cls.group_shape or a.shape == (cls.group_shape[0],)
    
    @classmethod
    def one(cls):
        return ca.SX.zeros(3, 1)
    
    @classmethod
    def product(cls, a, b):
        cls.check_group_shape(a)
        cls.check_group_shape(b)
        return ca.mtimes(a, b)

    @classmethod
    def inv(cls, a): #input a matrix of ca.SX form
        cls.check_group_shape(a)
        a_inv = ca.solve(a,ca.SX.eye(3)) #Google Group post mentioned ca.inv() could take too long, and should explore solve function
        return ca.transpose(a)
    
    @classmethod
    def log(cls, G):
        theta = ca.arccos(((G[0, 0]+G[1, 1]) - 1) / 2) #RECHECK (Unsure of where this comes from)
        wSkew = se2.wedge(G[:2,:2])

        # t is translational component vector
        t = ca.SX(3, 1)     
        t[0, 0] = G[0, 2]
        t[1, 0] = G[1, 2]
        
        if type(v[1]) == 'int' and theta < eps:
            a = 1 - theta ** 2 / 6 + theta ** 4 / 120
            b = 0.5 - theta ** 2 / 24 + theta ** 4 / 720
        else:
            a = ca.sin(theta)/theta
            b = (1-ca.cos(theta)/theta)

        V_inv = ca.SX(2,2)
        V_inv[0,0] = a
        V_inv[0,1] = b
        V_inv[1,0] = -b
        V_inv[1,1] = a
        V_inv = V_inv / (a**2 + b**2)


        vt_i= ca.mtimes(V_inv, t) 
        t_term = theta #Last Row for se2


        return ca.vertcat(vt_i, t_term)



def se2_diff_correction(v): #U Matrix for se2 with input vee operator
    return ca.inv(se2_diff_correction_inv(v))


def se2_diff_correction_inv(v): #U_inv of se2 input vee operator

    # v = se2.vee(v)  #This only applies if v is inputed from Lie Group format

    theta = v[2]
    # X_so3 = se2.wedge(v) #wedge operator for so2 (required [x,y,theta])

    if type(v[1]) == 'int' and theta < eps:
        c1 = 1 - theta ** 2 / 6 + theta ** 4 / 120
        c2 = 0.5 - theta ** 2 / 24 + theta ** 4 / 720
    else:
        c1 = ca.sin(theta)/theta
        c2 = (1-ca.cos(theta)/theta)

    ad = se2.ad_matrix(v)
    I = ca.SX_eye(3)
    u_inv = I + c1 * ad + c2 *se2.matmul(ad,ad)
    return u_inv

def fromSE2_dot_plot_draw(u, **kwargs):
    F = ca.sparsify(u)

    output_dir = '/home/wsribunm/Documents/GitHub/pyecca/pyecca_se3_implementation' #change directory if needed
    os.makedirs(output_dir, exist_ok=True)
    g = graph.dotgraph(F)
    g.set('dpi', 180)
    g.write_png(os.path.join(output_dir, 'se2_result_test_simplified.png'))
