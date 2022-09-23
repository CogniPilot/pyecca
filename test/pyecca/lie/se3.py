from . import so3
import casadi as ca

eps = 1e-7 # to avoid divide by zero


class SE3:
    '''
    Implementation of the mathematical group SE3, representing
    the 3D rigid body transformations
    '''
    
    group_shape = (4, 4)
    group_params = 12
    algebra_params = 6
    
    # coefficients 
    x = ca.SX.sym('x')
    C1 = ca.Function('a', [x], [ca.if_else(ca.fabs(x) < eps, 1 - x ** 2 / 6 + x ** 4 / 120, ca.sin(x)/x)])
    C2 = ca.Function('b', [x], [ca.if_else(ca.fabs(x) < eps, 0.5 - x ** 2 / 24 + x ** 4 / 720, (1 - ca.cos(x)) / x ** 2)])
    C3 = ca.Function('d', [x], [ca.if_else(ca.fabs(x) < eps, 0.5 + x**2/12 + 7*x**4/720, x/(2*ca.sin(x)))])
    C4 = ca.Function('f', [x], [ca.if_else(ca.fabs(x) < eps, (1/6) - x**2/120 + x**4/5040, (1-C1(x))/(x**2))])
    del x
    
    def __init__(self, SO3=None):
        if SO3 == None:
            self.SO3 = so3.Dcm()

    @classmethod
    def check_group_shape(cls, a):
        assert a.shape == cls.group_shape or a.shape == (cls.group_shape[0],)

    @classmethod
    def vee(cls, X):
        '''
        This takes in an element of the SE3 Lie Group and returns the se3 Lie Algebra elements 
        '''
        v = ca.SX(6, 1)
        v[0, 0] = X[2, 1]
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
        X = ca.SX.zeros(4, 4)
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
    def product(cls, a, b):
        cls.check_group_shape(a)
        cls.check_group_shape(b)
        return ca.mtimes(a, b)

    @classmethod
    def inv(cls, a):
        # TODO
        cls.check_group_shape(a)
        return ca.transpose(a)

    @classmethod
    def exp(cls, v):
        v = cls.vee(v)
        v_so3 = v[:3]
        X_so3 = so3.wedge(v_so3) 
        theta = ca.norm_2(so3.vee(X_so3))
        
        # translational components u
        u = ca.SX(3, 1)
        u[0, 0] = v[3]
        u[1, 0] = v[4]
        u[2, 0] = v[5]
        
        R = so3.Dcm.exp(v_so3)
        V = ca.SX.eye(3) + cls.C2(theta)*X_so3 + cls.C4(theta)*ca.mtimes(X_so3, X_so3)
        horz = ca.horzcat(R, ca.mtimes(V,u))
        
        lastRow = ca.horzcat(0,0,0,1)
        
        return ca.vertcat(horz, lastRow)

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

    @classmethod
    def kinematics(cls, R, w):
        # TODO
        assert R.shape == (3, 3)
        assert w.shape == (3, 1)
        return ca.mtimes(R, cls.wedge(w))
