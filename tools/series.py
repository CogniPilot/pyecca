"""
This is how the series expansions for SO3/SE3 were calculated.
"""

import sympy
sympy.init_printing()


t, t_f, omega = sympy.symbols('t, t_f, omega', positive=True)
theta = sympy.symbols('theta', positive=True)
A = sympy.sin(theta)/(theta)
B = (1 - sympy.cos(theta))/(theta)**2
C = (1 - A)/theta**2


print('A', A.series(theta))
print('B', B.series(theta))
print('C', C.series(theta))


alpha_int = sympy.Integral(B.subs(theta, omega*t)*t**2, (t, 0, t))
print('alpha integral', alpha_int)

alpha = alpha_int.doit().simplify().subs(omega, theta/t)
print('alpha', alpha)
print('alpha series', alpha.subs(t, 1).series(theta))

beta_int = sympy.Integral(C.subs(theta, omega*t)*t**3, (t, 0, t))
print('beta integral', beta_int)

beta = beta_int.doit().simplify().subs(omega, theta/t)
print('beta', beta)
print('beta series', beta.subs(t, 1).series(theta))

