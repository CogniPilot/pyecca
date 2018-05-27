# pyecca
**Py**thon **E**stimation and **C**ontrol library employing **Ca**sadi

The goal of Pyecca is to enable high level development of estimation and
control algorithms and rely on Casadi (casadi.org) for
generation of low level code for embedded control systems.

Due to the nature of the Casadi framework, an equation graph is
built for each algorithm. This equation graph represents an exact
mathematical equation and can be more easily analyzed than
normal C/C++/Python code. Consequently there is an obvious 
benefit to verification and validation of control systems.

For use in control systems, the casadi generated functions act as algorithmic
blocks that can be called. The method that data is sent to these blocks and
the timing should be handled separtely. For example, for an Extended Kalman
Filter, it will represent the predict and update methods without worrying
about how these are called. (see the [Examples](examples)).
