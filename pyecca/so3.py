import casadi as ca

eps = 0.001  # tolerance for switching to taylor series
x = ca.SX.sym('x')  # generic variable for functions

# coefficient functions, with taylor series approx near origin
a = ca.Function('a', [x], [ca.if_else(x < eps, 0.5 + x**2/12 + 7*x**4/720, x/(2*ca.sin(x)))])
b = ca.Function('b', [x], [ca.if_else(x < eps, 0.5 - x ** 2 / 24 + x ** 4 / 720, (1 - ca.cos(x)) / x ** 2)])


# noinspection PyPep8Naming
def vee(X):
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
def wedge(v):
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


# noinspection PyPep8Naming
def exp(X):
    """
    Exponential map from the Lie algebra to the Lie group
    :param X: A Lie algebra element
    :return: The corresponding Lie group element
    """
    omega = vee(X)
    theta = ca.sqrt(ca.dot(omega, omega))
    return ca.SX.eye(3) + a(theta)*X + b(theta)*ca.mtimes(X, X)


# noinspection PyPep8Naming
def log(R):
    """
    Inverse exponential map from the Lie group to the Lie algebra
    :param R: A Lie group rotation matrix element
    :return: The corresponding Lie algebra element to R
    """
    theta = ca.arccos((ca.trace(R) - 1)/2)
    return a(theta)*(R - R.T)


