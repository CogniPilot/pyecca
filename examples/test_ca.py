import casadi as ca

x = ca.SX.sym('x')

ca.if_else(x > 1, 1, 0)

f = ca.Function('test', [x], [ca.if_else(x > 1, 1, 0)], ['x'], ['res'])

print(f(2))