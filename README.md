# pyecca
Python Estimation and Control library employing CAsadi

The goal of Pyecca is to enable high level development of estimation and
control algorithms and rely on Casadi (casadi.org) for
generation of low level code for embedded control systems.

Due to the nature of the Casadi framework, an equation graph is
built for each algorithm. This equation graph represents an exact
mathematical equation and can be more easily analyzed than
normal C/C++/Python code. Consequently there is an obvious 
benefit to verification and validation of control systems.
