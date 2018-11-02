import casadi as ca


class DirectProduct:

    def __init__(self, groups):
        self.groups = groups
        self.n_group = [0]
        self.n_algebra = [0]
        shape_group_0 = 0
        shape_algebra_0 = 0
        for g in groups:
            ni = g.SHAPE[0] * g.SHAPE[1]
            self.n_group.append(ni)
            self.n_algebra.append(ni)
            shape_group_0 += ni
            shape_algebra_0 += ni
        self.group_shape = (shape_group_0, 1)
        self.algebra_shape = (shape_algebra_0, 1)

    def subgroup(self, a, i):
        start = 0
        for gi in range(i + 1):
            start += self.n_group[gi]
        end = start + self.n_group[i + 1]
        return a[start:end]

    def subalgebra(self, v, i):
        start = 0
        for gi in range(i):
            start += self.n_algebra[gi]
        end = start + self.n_algebra[i + 1]
        return v[start:end]

    def check_group_shape(self, a):
        assert a.shape == self.group_shape or a.shape == (self.group_shape[0],)

    def check_algebra_shape(self, a):
        assert a.shape == self.algebra_shape or a.shape == (self.algebra_shape[0],)

    def product(self, a, b):
        self.check_group_shape(a)
        self.check_group_shape(b)
        return ca.vertcat(*[
            g.product(self.subgroup(a, i), self.subgroup(b, i))
            for i, g in enumerate(self.groups)])

    def inv(self, a):
        self.check_group_shape(a)
        return ca.vertcat(*[
            g.inv(self.subgroup(a, i))
            for i, g in enumerate(self.groups)])

    def exp(self, v):
        self.check_algebra_shape(v)
        return ca.vertcat(*[
            g.exp(self.subalgebra(v, i))
            for i, g in enumerate(self.groups)])

    def log(self, a):
        self.check_group_shape(a)
        return ca.vertcat(*[
            g.log(self.subgroup(a, i))
            for i, g in enumerate(self.groups)])