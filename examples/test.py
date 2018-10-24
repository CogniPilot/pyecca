import casadi as ca
from examples.rotation import Mrp, Dcm, Quat, Euler

Expr = ca.SX

E = [
    ca.sparsify(Expr([
        [0, 0, 0],
        [0, 0, -1],
        [0, 1, 0]
    ])),
    ca.sparsify(Expr([
        [0, 0, 1],
        [0, 0, 0],
        [-1, 0, 0]
    ])),
    ca.sparsify(Expr([
        [0, -1, 0],
        [1, 0, 0],
        [0, 0, 0]
    ]))
]


def inner_product(A, B, W):
    return ca.trace(ca.mtimes([A, W, ca.transpose(B)]))


# inner product
def find_inner_product_matrix(E):
    n_d = E[0].shape[0]
    W_sym = ca.tril2symm(ca.SX.sym('W', ca.Sparsity_lower(n_d)))
    W_vect = ca.vertcat(*ca.symvar(W_sym))
    n_x = W_vect.shape[0]
    A = Expr(n_x, n_x)
    b = Expr(n_x, 1)
    cnt = 0
    for i in range(3):
        for j in range(i,3):
            res  = inner_product(E[i], E[j], W_sym)
            J = ca.jacobian(res, W_vect)
            assert len(ca.symvar(J)) == 0
            A[cnt, :] = J
            b[cnt] = (i == j)
            cnt += 1
    W_sol = ca.solve(A, b)
    W = ca.sparsify(ca.substitute(W_sym, W_vect, W_sol))
    return W


def param_data(E, g, q):
    W = find_inner_product_matrix(E)
    print('W', W)
    J_r = Expr(len(E), q.shape[0])
    J_l = Expr(len(E), q.shape[0])
    for i in range(len(E)):
        for j in range(q.shape[0]):
            J = ca.reshape(ca.jacobian(g, q[j]), g.shape[0], g.shape[1])
            g_inv = ca.inv(g)
            J_r[i, j] = inner_product(ca.mtimes(g_inv, J), E[i], W)
            J_l[i, j] = inner_product(ca.mtimes(J, g_inv), E[i], W)

    Ad = Expr(len(E), len(E))
    for i in range(len(E)):
        for j in range(len(E)):
            Ad[i, j] = inner_product(E[i], ca.mtimes([g, E[j], ca.inv(g)]), W)
    return J_r, J_l, Ad


def wedge(x, E):
    res = Expr(*E[0].shape)
    for i in range(len(E)):
        res += x[i]*E[i]
    return res


def vee(X, E):
    W = find_inner_product_matrix(E)
    res = Expr(len(E), 1)
    for i in range(len(E)):
        res[i] = inner_product(X, E[i], W)
    return res


e1 = Euler([0.1, 0.2, 0.3])
q1 = Quat.from_euler(e1)
r1 = Mrp.from_euler(e1)

q = Quat(ca.SX.sym('q', 4, 1))
x = ca.SX.sym('x', 3, 1)
x1 = ca.SX([1, 2, 3])
X = wedge(x, E)
g = Dcm.from_quat(q)
J_r, J_l, Ad = param_data(E, g, q)

print('wedge', wedge(x1, E))
print('vee', vee(wedge(x1, E), E))

print('g', ca.substitute(g, q, q1))
print('Ad', ca.substitute(Ad, q, q1))
print('Ad(g) * x1', ca.substitute(ca.mtimes(Ad, x1), q, q1))
print('(g * x1^ * inv(g))v', ca.substitute(vee(ca.mtimes([g, wedge(x1, E), ca.inv(g)]), E), q, q1))

#print('g * g', ca.substitute(ca.mtimes(g, g), q, q1))

print(Ad, g)
#r = Mrp(Expr.sym('r', 3))
#g = Dcm.from_mrp(r)

#ca.mtimes(ca.inv(g), ca.reshape(ca.jacobian(g, r[0]), 3, 3))