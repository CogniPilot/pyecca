"""
A module for Direction Cosine Matrices (DCMs).

This is the standard representation of SO(3). There are 9 parameters and no singularities.
"""
import casadi as ca

Expr = ca.SX


# coefficient functions, with taylor series approx near origin
eps = 0.001
x = Expr.sym('x')
C1 = ca.Function('a', [x], [ca.if_else(x < eps, 1 - x ** 2 / 6 + x ** 4 / 120, ca.sin(x)/x)])
C2 = ca.Function('b', [x], [ca.if_else(x < eps, 0.5 - x ** 2 / 24 + x ** 4 / 720, (1 - ca.cos(x)) / x ** 2)])
C3 = ca.Function('d', [x], [ca.if_else(x < eps, 0.5 + x**2/12 + 7*x**4/720, x/(2*ca.sin(x)))])


class Dcm(Expr):

    def __init__(self, *args):
        super().__init__(*args)
        assert self.shape == (3, 3)

    def __rmul__(self, other: Expr) -> 'Dcm':
        assert other.shape == (1, 1)
        return Dcm(other * Expr(self))

    def __mul__(self, other: Expr) -> 'Dcm':
        if other.shape == (3, 3):
            return Dcm(ca.mtimes(self, other))
        else:
            return ca.mtimes(self, other)

    def inv(self) -> 'Dcm':
        return Dcm(ca.inv(self))

    @classmethod
    def exp(cls, w: Expr) -> 'Dcm':
        """
        The exponential map from the Lie algebra element components to the Lie group.
        :param w: The Lie algebra, represented by components of an angular velocity vector.
        :return: The Lie group, represented by a DCM.
        """
        theta = ca.norm_2(w)
        X = cls.wedge(w)
        return Dcm(ca.SX.eye(3) + C1(theta)*X + C2(theta)*ca.mtimes(X, X))

    def log(self) -> Expr:
        """
        The inverse exponential map form the Lie group to the Lie algebra element components.
        :return: The Lie algebra, represented by an angular velocity vector.
        """
        theta = ca.arccos((ca.trace(self) - 1) / 2)
        return self.vee(C3(theta) * (self - self.T))

    def derivative(self, w: Expr) -> Expr:
        """
        The kinematic equation relating the time derivative of MRPs given the current MRPs and the angular velocity.
        :param w: The angular velocity.
        :return: The time derivative of the DCM.
        """
        return ca.mtimes(self, Dcm.wedge(w))

    # noinspection PyPep8Naming
    @classmethod
    def vee(cls, X):
        """
        Takes a Lie algebra element and extracts components
        :param X: Lie algebra element
        :return:
        """
        v = ca.SX(3, 1)
        v[0, 0] = X[2, 1]
        v[1, 0] = X[0, 2]
        v[2, 0] = X[1, 0]
        return v


    # noinspection PyPep8Naming
    @classmethod
    def wedge(cls, v):
        """
        Take Lie algebra components and builds a Lie algebra element.
        :param v:
        :return:
        """
        X = ca.SX(3, 3)
        X[0, 1] = -v[2]
        X[0, 2] = v[1]
        X[1, 0] = v[2]
        X[1, 2] = -v[0]
        X[2, 0] = -v[1]
        X[2, 1] = v[0]
        return X