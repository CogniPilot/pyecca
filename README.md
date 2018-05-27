# pyecca
Python Estimation and Control library employing CAsadi

This is a library for estimation and control that employs
Casadi (https://github.com/casadi/casadi/wiki), a symbolic
framework for algorithm differentiation.

The goal is to allow high level development of control and
estimation algorithms in this framework and then use Casadi to
generate low level code for embedded control systems.

Due to the nature of the Casadi framework, an equation graph is
built for each algorithm. This equation graph represents an exact
mathematical equation and can be more easily analyzed than
normal C/C++/Python code. Consequently there is an obvious 
benefit to verification and validation of control systems.

