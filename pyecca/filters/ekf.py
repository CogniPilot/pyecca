import casadi as ca


# noinspection PyPep8Naming
class Ekf:

    def __init__(self, expr, n_x, n_u):
        self.n_x = n_x
        self.n_u = n_u
        self.expr = expr
        self.x = expr.sym('x', n_x)
        self.u = expr.sym('u', n_u)
        self.PU = expr.sym('P', ca.Sparsity.upper(n_x))
        self.sigma_w = expr.sym('sigma_w', n_x)
        self.Q = ca.diag(self.sigma_w)

    def state_derivative(self, name, f):
        dx = ca.simplify(f(self.expr, self.x, self.u))
        return ca.Function(name, [self.x, self.u], [dx], ['x', 'u'], ['dx'])

    def covariance_derivative(self, name, f):
        F = ca.jacobian(f(self.expr, self.x, self.u), self.x)
        # G = ca.jacobian(f(self.expr, self.x, self.u), self.u)
        P = ca.triu2symm(self.PU)
        dP = ca.mtimes(F, P) + ca.mtimes(P, F.T) + self.Q

        # force symmetric
        dP = ca.simplify(ca.triu2symm(ca.triu(dP)))
        return ca.Function(name, [self.x, self.u, self.PU, self.sigma_w], [dP],
                           ['x', 'u', 'PU', 'sigma_w'], ['dP'])

    def correct(self, name, g):
        yv = g(self.expr, self.x)
        n_y = yv.shape[0]
        y = self.expr.sym('y', n_y)
        sigma_v = self.expr.sym('sigma_v', n_y)
        R = ca.diag(sigma_v ** 2)
        P = ca.triu2symm(self.PU)
        H = ca.jacobian(g(self.expr, self.x), self.x)
        S = ca.mtimes([H, P, H.T]) + R
        K = ca.mtimes([P, H.T, ca.inv(S)])
        P1 = ca.mtimes((self.expr.eye(self.n_x) - ca.mtimes(K, H)), P)
        # force symmetric
        P1 = ca.simplify(ca.triu2symm(ca.triu(P1)))
        r = y - ca.mtimes(H, self.x)
        x1 = ca.simplify(self.x + ca.mtimes(K, r))
        return ca.Function(name, [self.x, y, self.PU, sigma_v], [x1, P1], ['x', 'y', 'PU', 'sigma_v'], ['x1', 'P1'])
