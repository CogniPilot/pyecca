import casadi as ca


def product(a, b):
    assert a.shape == (4, 1) or a.shape == (4,)
    assert b.shape == (4, 1) or b.shape == (4,)
    r1 = a[0]
    v1 = a[1:]
    r2 = b[0]
    v2 = b[1:]
    res = ca.SX(4, 1)
    res[0] = r1 * r2 - ca.dot(v1, v2)
    res[1:] = r1 * v2 + r2 * v1 + ca.cross(v1, v2)
    return res


def inv(q):
    assert q.shape == (4, 1) or q.shape == (4,)
    qi = ca.SX(4, 1)
    n = ca.norm_2(q)
    qi[0] = q[0]/n
    qi[1] = -q[1]/n
    qi[2] = -q[2]/n
    qi[3] = -q[3]/n
    return ca.if_else(n > 0, qi, ca.DM([1, 0, 0, 0]))


def exp(v):
    assert v.shape == (3, 1) or q.shape == (3,)
    q = ca.SX(4, 1)
    theta = ca.norm_2(v)
    q[0] = ca.cos(theta/2)
    c = ca.sin(theta/2)
    n = ca.norm_2(v)
    q[1] = c*v[0]/n
    q[2] = c*v[1]/n
    q[3] = c*v[2]/n
    return ca.if_else(n > 0, q, ca.DM([1, 0, 0, 0]))


def log(q):
    assert q.shape == (4, 1) or q.shape == (4,)
    v = ca.SX(3, 1)
    norm_q = ca.norm_2(q)
    theta = 2*ca.acos(q[0])
    c = ca.sin(theta/2)
    v[0] = theta*q[1]/c
    v[1] = theta*q[2]/c
    v[2] = theta*q[3]/c
    return ca.if_else(ca.logic_or(ca.fabs(q[0]) >= 1, norm_q == 0), ca.SX([0, 0, 0]), v)


def kinematics(q, w):
    """
    The kinematic equation relating the time derivative of quat given the current quat and the angular velocity
    in the body frame.
    :param w: The angular velocity in the body frame.
    :return: The time derivative of the quat.
    """
    assert q.shape == (4, 1) or q.shape == (4,)
    assert w.shape == (3, 1) or w.shape == (3,)
    v = ca.SX(4, 1)
    v[0] = 0
    v[1] = w[0]
    v[2] = w[1]
    v[3] = w[2]
    return 0.5 * product(q, v)


def from_mrp(r):
    assert r.shape == (4, 1) or r.shape == (4,)
    a = r[:3]
    q = ca.SX(4, 1)
    n_sq = ca.dot(a, a)
    den = 1 + n_sq
    q[0] = (1 - n_sq)/den
    for i in range(3):
        q[i + 1] = 2*a[i]/den
    res = ca.if_else(den > 0, q, ca.SX([1, 0, 0, 0]))
    return ca.if_else(r[3], -res, res)


def from_dcm(R):
    assert R.shape == (3, 3)
    b1 = 0.5 * ca.sqrt(1 + R[0, 0] + R[1, 1] + R[2, 2])
    b2 = 0.5 * ca.sqrt(1 + R[0, 0] - R[1, 1] - R[2, 2])
    b3 = 0.5 * ca.sqrt(1 - R[0, 0] + R[1, 1] - R[2, 2])
    b4 = 0.5 * ca.sqrt(1 - R[0, 0] - R[1, 1] - R[2, 2])

    q1 = ca.SX(4, 1)
    q1[0] = b1
    q1[1] = (R[2, 1] - R[1, 2]) / (4 * b1)
    q1[2] = (R[0, 2] - R[2, 0]) / (4 * b1)
    q1[3] = (R[1, 0] - R[0, 1]) / (4 * b1)

    q2 = ca.SX(4, 1)
    q2[0] = (R[2, 1] - R[1, 2]) / (4 * b2)
    q2[1] = b2
    q2[2] = (R[0, 1] + R[1, 0]) / (4 * b2)
    q2[3] = (R[0, 2] + R[2, 0]) / (4 * b2)

    q3 = ca.SX(4, 1)
    q3[0] = (R[0, 2] - R[2, 0]) / (4 * b3)
    q3[1] = (R[0, 1] + R[1, 0]) / (4 * b3)
    q3[2] = b3
    q3[3] = (R[1, 2] + R[2, 1]) / (4 * b3)

    q4 = ca.SX(4, 1)
    q4[0] = (R[1, 0] - R[0, 1]) / (4 * b4)
    q4[1] = (R[0, 2] + R[2, 0]) / (4 * b4)
    q4[2] = (R[1, 2] + R[2, 1]) / (4 * b4)
    q4[3] = b4

    q = ca.if_else(
        R[0, 0] > 0,
        ca.if_else(R[1, 1] > 0, q1, q2),
        ca.if_else(R[1, 1] > R[2, 2], q3, q4)
    )
    return q


def from_euler(e):
    assert e.shape == (3, 1) or e.shape == (3,)
    q = ca.SX(4, 1)
    cosPhi_2 = ca.cos(e[0]/2)
    cosTheta_2 = ca.cos(e[1]/2)
    cosPsi_2 = ca.cos(e[2]/2)
    sinPhi_2 = ca.sin(e[0]/2)
    sinTheta_2 = ca.sin(e[1]/2)
    sinPsi_2 = ca.sin(e[2]/2)
    q[0] = cosPhi_2 * cosTheta_2 * cosPsi_2 + \
           sinPhi_2 * sinTheta_2 * sinPsi_2
    q[1] = sinPhi_2 * cosTheta_2 * cosPsi_2 - \
           cosPhi_2 * sinTheta_2 * sinPsi_2
    q[2] = cosPhi_2 * sinTheta_2 * cosPsi_2 + \
           sinPhi_2 * cosTheta_2 * sinPsi_2
    q[3] = cosPhi_2 * cosTheta_2 * sinPsi_2 - \
           sinPhi_2 * sinTheta_2 * cosPsi_2
    return q
