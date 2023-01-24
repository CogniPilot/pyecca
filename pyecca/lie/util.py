import casadi as ca

eps = 1e-7  # to avoid divide by zero

x = ca.SX.sym("x")

# sin(x)/x
C1 = ca.Function(
    "a",
    [x],
    [ca.if_else(ca.fabs(x) < eps, 1 - x**2 / 6 + x**4 / 120, ca.sin(x) / x)],
)

# (1 - cos(x))/x^2
C2 = ca.Function(
    "b",
    [x],
    [
        ca.if_else(
            ca.fabs(x) < eps,
            0.5 - x**2 / 24 + x**4 / 720,
            (1 - ca.cos(x)) / x**2,
        )
    ],
)

# x/ (2 sin(x))
C3 = ca.Function(
    "d",
    [x],
    [
        ca.if_else(
            ca.fabs(x) < eps,
            0.5 + x**2 / 12 + 7 * x**4 / 720,
            x / (2 * ca.sin(x)),
        )
    ],
)

# (1 - C1)/x^2
C4 = ca.Function(
    "f",
    [x],
    [
        ca.if_else(
            ca.fabs(x) < eps,
            (1 / 6) - x**2 / 120 + x**4 / 5040,
            (1 - C1(x)) / (x**2),
        )
    ],
)
    
# delete temp variable used to create functions
del x
