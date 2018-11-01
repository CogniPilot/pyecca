import casadi as ca


def test():
    x = ca.SX.sym('x')
    y = ca.SX.sym('y')

    f_inv = ca.Function('inv', [x], [2*x])
    f_mul = ca.Function('mul', [x, y], [ca.mtimes(x, y)])
    f_exp = ca.Function('exp', [x], [ca.exp(x)])
    f_log = ca.Function('exp', [x], [ca.log(x)])

    so3 = {
        'inv': f_inv,
        'mul': f_mul,
        'exp': f_exp,
        'log': f_log
    }

    so3['mul'](1, 2)


    f_test = f_inv(x)
    print('f_test', f_test)


    def direct_product(groups):
        f_inv = ca.Function('inv', [])

    print('group', so3)


