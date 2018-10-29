import casadi as ca


Expr = ca.SX
eps = 1e-8 # tolerance for avoiding divide by 0


class Dcm(Expr):

    def __init__(self, *args):
        super().__init__(*args)
        assert self.shape == (3, 3)

    @classmethod
    def from_quat(cls, q: 'Quat'):
        """
        Converts a quaternion to a DCM.
        :return: The DCM.
        """
        assert isinstance(q, Quat)
        R = Expr(3, 3)
        a = q[0]
        b = q[1]
        c = q[2]
        d = q[3]
        aa = a * a
        ab = a * b
        ac = a * c
        ad = a * d
        bb = b * b
        bc = b * c
        bd = b * d
        cc = c * c
        cd = c * d
        dd = d * d
        R[0, 0] = aa + bb - cc - dd
        R[0, 1] = 2 * (bc - ad)
        R[0, 2] = 2 * (bd + ac)
        R[1, 0] = 2 * (bc + ad)
        R[1, 1] = aa + cc - bb - dd
        R[1, 2] = 2 * (cd - ab)
        R[2, 0] = 2 * (bd - ac)
        R[2, 1] = 2 * (cd + ab)
        R[2, 2] = aa + dd - bb - cc
        return cls(R)

    @classmethod
    def from_mrp(cls, r: 'Mrp') -> 'Dcm':
        """
        Converts a Mrp to a Dcm.
        :return: The Dcm.
        """
        assert isinstance(r, Mrp)
        return cls.from_quat(Quat.from_mrp(r))

    @classmethod
    def from_euler(cls, e: 'Euler') -> 'Dcm':
        """
        Convert body 321 euler angles to a Dcm.
        :return: The Dcm.
        """
        assert isinstance(e, Euler)
        cosPhi = ca.cos(e.phi)
        sinPhi = ca.sin(e.phi)
        cosThe = ca.cos(e.theta)
        sinThe = ca.sin(e.theta)
        cosPsi = ca.cos(e.psi)
        sinPsi = ca.sin(e.psi)

        R = Expr(3, 3)
        R[0, 0] = cosThe * cosPsi
        R[0, 1] = -cosPhi * sinPsi + sinPhi * sinThe * cosPsi
        R[0, 2] = sinPhi * sinPsi + cosPhi * sinThe * cosPsi

        R[1, 0] = cosThe * sinPsi
        R[1, 1] = cosPhi * cosPsi + sinPhi * sinThe * sinPsi
        R[1, 2] = -sinPhi * cosPsi + cosPhi * sinThe * sinPsi

        R[2, 0] = -sinThe
        R[2, 1] = sinPhi * cosThe
        R[2, 2] = cosPhi * cosThe

        return cls(R)


class Quat(Expr):

    def __init__(self, *args):
        super().__init__(*args)
        assert self.shape == (4, 1)

    def __add__(self, other: 'Quat') -> 'Quat':
        """
        Adds two quaternions element by element, should be avoided in general,
        resultant quaternion will need to be renormalized.
        :param other:
        :return:
        """
        return Quat(Expr(self) + Expr(other))

    def __sub__(self, other: 'Quat') -> 'Quat':
        """
        Subtract two quaternions element by element, should be avoided in
        general, resultant quaternion will need to be renormalized.
        :param other: quaternion to subtract
        :return: result
        """
        return Quat(Expr(self) - Expr(other))

    def __neg__(self):
        """
        Take the negative element by element of a quaternion. Gives the
        shadow quaternion, which represents the same orientation. Generally no
        need to do this.
        :return: result
        """
        return Quat(-Expr(self))

    def __rmul__(self, other: Expr) -> 'Quat':
        """
        Multiply by a scalar.
        :param other: scalar
        :return:
        """
        s = Expr(other)
        assert s.shape == (1, 1)
        return Quat(other * Expr(self))

    def __mul__(self, other: 'Quat') -> 'Quat':
        """
        The product of two quaternions using the hamilton
        convention, so that Dcm(A)*Dcm(B) = Dcm(A*B).
        :param other: The second quaternion.
        :return: The quaternion product.
        """
        assert isinstance(other, Quat)
        a = self
        b = other
        r1 = a[0]
        v1 = a[1:]
        r2 = b[0]
        v2 = b[1:]
        res = Expr(4, 1)
        res[0] = r1 * r2 - ca.dot(v1, v2)
        res[1:] = r1 * v2 + r2 * v1 + ca.cross(v1, v2)
        return Quat(res)

    def derivative(self, w: Expr) -> Expr:
        """
        The kinematic equation relating the time derivative of quat given the current quat and the angular velocity
        in the body frame.
        :param w: The angular velocity in the body frame.
        :return: The time derivative of the quat.
        """
        v = Expr(4, 1)
        v[0] = 0
        v[1] = w[0]
        v[2] = w[1]
        v[3] = w[2]
        return 0.5 * self * Quat(v)

    @classmethod
    def from_dcm(cls, R: Dcm) -> Expr:
        """
        Converts a direction cosine matrix to a quaternion.
        :param R: A direction cosine matrix.
        :return: The quaternion.
        """
        b1 = 0.5 * ca.sqrt(1 + R[0, 0] + R[1, 1] + R[2, 2])
        b2 = 0.5 * ca.sqrt(1 + R[0, 0] - R[1, 1] - R[2, 2])
        b3 = 0.5 * ca.sqrt(1 - R[0, 0] + R[1, 1] - R[2, 2])
        b4 = 0.5 * ca.sqrt(1 - R[0, 0] - R[1, 1] - R[2, 2])

        q1 = Expr(4, 1)
        q1[0] = b1
        q1[1] = (R[2, 1] - R[1, 2]) / (4 * b1)
        q1[2] = (R[0, 2] - R[2, 0]) / (4 * b1)
        q1[3] = (R[1, 0] - R[0, 1]) / (4 * b1)

        q2 = Expr(4, 1)
        q2[0] = (R[2, 1] - R[1, 2]) / (4 * b2)
        q2[1] = b2
        q2[2] = (R[0, 1] + R[1, 0]) / (4 * b2)
        q2[3] = (R[0, 2] + R[2, 0]) / (4 * b2)

        q3 = Expr(4, 1)
        q3[0] = (R[0, 2] - R[2, 0]) / (4 * b3)
        q3[1] = (R[0, 1] + R[1, 0]) / (4 * b3)
        q3[2] = b3
        q3[3] = (R[1, 2] + R[2, 1]) / (4 * b3)

        q4 = Expr(4, 1)
        q4[0] = (R[1, 0] - R[0, 1]) / (4 * b4)
        q4[1] = (R[0, 2] + R[2, 0]) / (4 * b4)
        q4[2] = (R[1, 2] + R[2, 1]) / (4 * b4)
        q4[3] = b4

        q = ca.if_else(
            R[0, 0] > 0,
            ca.if_else(R[1, 1] > 0, q1, q2),
            ca.if_else(R[1, 1] > R[2, 2], q3, q4)
        )
        return cls(q)

    @classmethod
    def from_euler(cls, e: 'Euler') -> 'Quat':
        assert isinstance(e, Euler)
        cosPhi_2 = ca.cos(e.phi / 2)
        cosTheta_2 = ca.cos(e.theta / 2)
        cosPsi_2 = ca.cos(e.psi / 2)
        sinPhi_2 = ca.sin(e.phi / 2)
        sinTheta_2 = ca.sin(e.theta / 2)
        sinPsi_2 = ca.sin(e.psi / 2)
        q = Expr(4, 1)
        q[0] = cosPhi_2 * cosTheta_2 * cosPsi_2 + sinPhi_2 * sinTheta_2 * sinPsi_2
        q[1] = sinPhi_2 * cosTheta_2 * cosPsi_2 - cosPhi_2 * sinTheta_2 * sinPsi_2
        q[2] = cosPhi_2 * sinTheta_2 * cosPsi_2 + sinPhi_2 * cosTheta_2 * sinPsi_2
        q[3] = cosPhi_2 * cosTheta_2 * sinPsi_2 - sinPhi_2 * sinTheta_2 * cosPsi_2
        return cls(q)

    @classmethod
    def from_mrp(cls, r: 'Mrp') -> 'Quat':
        assert isinstance(r, Mrp)
        a = r[0:3]
        q = Expr(4, 1)
        n_sq = ca.dot(a, a)
        den = 1 + n_sq
        q[0] = (1 - n_sq)/den
        for i in range(3):
            q[i + 1] = 2*a[i]/den
        q = ca.if_else(r[3] == 0, q, -q)
        return cls(q)


class Mrp(Expr):

    def __init__(self, *args):
        super().__init__(*args)
        assert self.shape == (4, 1)

    def __add__(self, other: 'Mrp') -> 'Mrp':
        """
        Adds two MRPs element by element, should be avoided in general.
        :param other:
        :return:
        """
        return Quat(Expr(self) + Expr(other))

    def __sub__(self, other: 'Mrp') -> 'Mrp':
        """
        Subtract two MRPs element by element, should be avoided in
        general.
        :param other: quaternion to subtract
        :return: result
        """
        return Mrp(Expr(self) - Expr(other))

    def __neg__(self):
        """
        Take the negative element by element of a Mrp.
        :return: result
        """
        return Mrp(-Expr(self))

    def B(self) -> Expr:
        """
        A matrix used to compute the MRPs kinematics.
        :return: The B matrix.
        """
        a = self[0:3]
        n_sq = ca.dot(a, a)
        X = so3.wedge(a)
        return 0.25 * ((1 - n_sq) * Expr.eye(3) + 2 * X + 2 * ca.mtimes(a, ca.transpose(a)))

    def derivative(self, w: Expr) -> Expr:
        """
        The kinematic equation relating the time derivative of MRPs given the current MRPs and the angular velocity
        in the body frame.
        :param w: The angular velocity in the body frame.
        :return: The time derivative of the MRPs.
        """
        return ca.vertcat(ca.mtimes(self.B(), w), 0)

    def shadow(self) -> 'Mrp':
        """
        Convert MRPs to their shadow (the MRPs corresponding to the quaternion with opposite sign). Both the MRPs and
        shadow MRPs represent the same attitude, but one of the two's magnitude is always less than 1, while the other's
        magnitude can approach inf near a rotation of 2*pi. So this function is used to switch to the other set when an
        MRP magnitude is greater than one to avoid the singularity.
        :return: The shadow MRP
        """
        a = self[0:3]
        n_sq = ca.dot(a, a)
        s = ca.if_else(self[3], 0, 1)  # toggle shadow state
        return Mrp(ca.if_else(n_sq > eps, ca.vertcat(-a / n_sq, s), [0, 0, 0, 0]))

    def shadow_if_required(self):
        """
        Performs the shadowing operation if required
        :return: The MRP after possible shadowing
        """
        return Mrp(ca.if_else(ca.norm_fro(self[:3]) > 1, self.shadow(), self))

    @classmethod
    def from_quat(cls, q: Quat) -> 'Mrp':
        """
        Convert from a quaternion to MRPs.
        :param q: The quaternion.
        :return: The MRPs.
        """
        assert isinstance(q, Quat)
        den = 1 + q[0]
        r = Expr(4, 1)
        r[0] = q[1] / den
        r[1] = q[2] / den
        r[2] = q[3] / den
        r[3] = 0
        return cls(r)

    @classmethod
    def from_euler(cls, e: 'Euler') -> 'Mrp':
        assert isinstance(e, Euler)
        return cls.from_quat(Quat.from_euler(e))

    @classmethod
    def from_dcm(cls, R: 'Dcm') -> 'Mrp':
        assert isinstance(R, Dcm)
        return cls.from_quat(Quat.from_dcm(R))

class Euler(Expr):

    def __init__(self, *args):
        super().__init__(*args)
        assert self.shape == (3, 1)

    @property
    def phi(self):
        return self[0]

    @property
    def theta(self):
        return self[1]

    @property
    def psi(self):
        return self[2]

    @classmethod
    def from_quat(cls, q: Quat) -> 'Euler':
        """
        Converts a quaternion to B321 Euler angles.
        :return: The B321 Euler angles (Phi [roll], Theta [pitch], Psi[heading])
        """
        return cls.from_dcm(Dcm.from_quat(q))

    @classmethod
    def from_dcm(cls, R: Dcm) -> 'Euler':
        phi = ca.atan2(R[2, 1], R[2, 2])
        theta = ca.asin(-R[2, 0])
        psi = ca.atan2(R[1, 0], R[0, 0])
        e = Expr(3, 1)
        e[0] = ca.if_else(
            ca.logic_or(ca.fabs(theta - ca.pi / 2) < eps, ca.fabs(theta + ca.pi / 2) < eps),
            0, phi)
        e[1] = theta
        e[2] = ca.if_else(
            ca.fabs(theta - ca.pi / 2) < eps,
            ca.atan2(R[1, 2], R[0, 2]),
            ca.if_else(
                ca.fabs(theta + ca.pi / 2) < eps,
                ca.atan2(-R[1, 2], -R[0, 2]),
                psi))
        return cls(e)

    @classmethod
    def from_mrp(cls, r: Mrp) -> 'Euler':
        return cls.from_quat(Quat.from_mrp(r))

e1 = Euler([0.1, 0.2, 0.3])
q1 = Quat.from_euler(e1)
r1 = Mrp.from_euler(e1)

# test round trip conversions (DCM, Quat, Mrp, Euler), 6 total , 4*3/2
assert ca.norm_fro(Euler.from_quat(Quat.from_euler(e1)) - e1) < 1e-6
assert ca.norm_fro(Euler.from_dcm(Dcm.from_euler(e1)) - e1) < 1e-6
assert ca.norm_fro(Euler.from_mrp(Mrp.from_euler(e1)) - e1) < 1e-6
assert ca.norm_fro(Quat.from_dcm(Dcm.from_quat(q1)) - q1) < 1e-6
assert ca.norm_fro(Quat.from_mrp(Mrp.from_quat(q1)) - q1) < 1e-6
assert ca.norm_fro(Mrp.from_dcm(Dcm.from_mrp(r1)) - r1) < 1e-6

#r = Mrp(ca.SX.sym('r', 3))
#f = ca.Function('f', [r], [Euler.from_mrp(r).psi], ['r'], ['psi'])
#J = ca.Function('J', [r], [ca.jacobian(f(r), r)])


# coefficient functions, with taylor series approx near origin
x = Expr.sym('x')
C1 = ca.Function('a', [x], [ca.if_else(x < eps, 1 - x ** 2 / 6 + x ** 4 / 120, ca.sin(x)/x)])
C2 = ca.Function('b', [x], [ca.if_else(x < eps, 0.5 - x ** 2 / 24 + x ** 4 / 720, (1 - ca.cos(x)) / x ** 2)])
C3 = ca.Function('d', [x], [ca.if_else(x < eps, 0.5 + x**2/12 + 7*x**4/720, x/(2*ca.sin(x)))])


class SO3(Expr):
    @classmethod
    def exp(cls, w: Expr) -> 'Dcm':
        """
        The exponential map from the Lie algebra element components to the Lie group.
        :param w: The Lie algebra, represented by components of an angular velocity vector.
        :return: The Lie group, represented by a DCM.
        """
        theta = ca.norm_2(w)
        X = cls.wedge(w)
        return Dcm(ca.SX.eye(3) + C1(theta)*X + C2(theta)*ca.mtimes(X, X))

    def log(self) -> Expr:
        """
        The inverse exponential map form the Lie group to the Lie algebra element components.
        :return: The Lie algebra, represented by an angular velocity vector.
        """
        theta = ca.arccos((ca.trace(self) - 1) / 2)
        return self.vee(C3(theta) * (self - self.T))

    # noinspection PyPep8Naming
    @classmethod
    def vee(cls, X: Expr) -> Expr:
        """
        Takes a Lie algebra element and extracts components
        :param X: Lie algebra element
        :return:
        """
        v = ca.SX(3, 1)
        v[0, 0] = X[2, 1]
        v[1, 0] = X[0, 2]
        v[2, 0] = X[1, 0]
        return v


    # noinspection PyPep8Naming
    @classmethod
    def wedge(cls, v: Expr) -> Expr:
        """
        Take Lie algebra components and builds a Lie algebra element.
        :param v:
        :return:
        """
        X = ca.SX(3, 3)
        X[0, 1] = -v[2]
        X[0, 2] = v[1]
        X[1, 0] = v[2]
        X[1, 2] = -v[0]
        X[2, 0] = -v[1]
        X[2, 1] = v[0]
        return X


class so3(Expr):

    @classmethod
    def wedge(cls, v):
        X = Expr(3, 3)
        X[0, 1] = -v[2]
        X[1, 0] = v[2]
        X[0, 2] = v[1]
        X[2, 0] = -v[1]
        X[1, 2] = -v[0]
        X[2, 1] = v[0]
        return X

R = Dcm(ca.SX.sym('R', 3, 3))
r = Mrp(ca.SX.sym('r', 4, 1))
R_r = Dcm.from_mrp(r)
omega = ca.SX.sym('omega', 3)
res = ca.mtimes(ca.jacobian(r.from_dcm(R), R), ca.reshape(ca.mtimes(R, so3.wedge(omega)), 9, 1))