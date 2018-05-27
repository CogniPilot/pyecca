"""
This package contains various filtering algorithms.

ekf: Extended Kalman Filter - Linearizes measurements and dynamics, then uses standard Kalman Filter
iekf: Invariant Extended Kalman Filter - Uses Lie group log/exp for correction term.
ukf: Unscented Kalman Filter - Uses sigma points to approximate PDF
"""