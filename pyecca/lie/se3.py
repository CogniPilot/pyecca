import so3

import casadi as ca

eps = 1e-7 # to avoid divide by zero


def vee(X):
    v = ca.SX(6, 1)
    v[0, 0] = X[2, 1]
    v[1, 0] = X[0, 2]
    v[2, 0] = X[1, 0]
    v[3, 0] = X[3, 0]
    v[4, 0] = X[3, 1]
    v[5, 0] = X[3, 2]
    return v


def wedge(v):
    X = ca.SX.zeros(4, 4)
    X[0, 1] = -v[2]
    X[0, 2] = v[1]
    X[1, 0] = v[2]
    X[1, 2] = -v[0]
    X[2, 0] = -v[1]
    X[2, 1] = v[0]
    X[3, 0] = v[3]
    X[3, 1] = v[4]
    X[3, 2] = v[5]
    return X


class SE3Dcm:

    group_shape = (4, 4)
    group_params = 12
    algebra_params = 6

    x = ca.SX.sym('x')
    C1 = ca.Function('a', [x], [ca.if_else(ca.fabs(x) < eps, 1 - x ** 2 / 6 + x ** 4 / 120, ca.sin(x)/x)])
    C4 = ca.Function('f', [x], [ca.if_else(ca.fabs(x) < eps, (1/6) - x**2/120 + x**4/5040, (1-C1(x))/(x**2))])
    
    def __init__(self):
        raise RuntimeError('this class is just for scoping, do not instantiate')
        if SO3 == None:
            self.SO3 = so3.DCM()

    @classmethod
    def check_group_shape(cls, a):
        assert a.shape == cls.group_shape or a.shape == (cls.group_shape[0],)

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
        # TODO
        X = wedge(v)
        theta = ca.norm_2(v)
        
        u = ca.SX(3, 1)
        u[0, 0] = v[3, 0]
        u[1, 0] = v[4, 0]
        u[2, 0] = v[5, 0]
        
        R = self.SO3.exp(v)
        V = ca.SX.eye(3) + so3.C2(theta)*X + cls.C4(theta)*ca.mtimes(X, X)
        
        vert = ca.vertcat(R, V*u)
        lastRow = ca.SX([0,0,0,1])
        
        return ca.horizcat(vert, lastRow)

    @classmethod
    def log(cls, R):
        # TODO
        theta = ca.arccos((ca.trace(R) - 1) / 2)
        wSkew = self.SO3.vee(cls.C3(theta) * (R - R.T))
        
        V_inv = ca.SX.eye(3) - 0.5*wSkew + (1/(theta**2))*(1-((so3.C1(theta))/(2*so3.C2(theta))))*ca.mtimes(wSkew, wSkew)
        
        # t is the translational component vector
        t = ca.SX(3, 1)     
        t[0, 0] = X[3, 0]
        t[1, 0] = X[3, 1]
        t[2, 0] = X[3, 2]
        
        uInv = ca.mtimes(V_inv * t) 
        vert2 = ca.vertcat(wSkew, uInv)
        lastRow2 = ca.SX([0,0,0,0])
        
        return ca.horizcat(vert2, lastRow2)

    @classmethod
    def kinematics(cls, R, w):
        # TODO
        assert R.shape == (3, 3)
        assert w.shape == (3, 1)
        return ca.mtimes(R, wedge(w))

# testing exp and log maps
v = [0.523, 0.785, 1.0467, 1, 2, 3]
print(vee(wedge(v))) # gives back v
# currently working on this, will have to check norm_2 function
print(vee(SE3Dcm.log(SE3Dcm.exp(wedge(v)))))