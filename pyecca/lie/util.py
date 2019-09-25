import casadi as ca


class DirectProduct:

    def __init__(self, groups):
        self.groups = groups
        self.n_group = [0]
        self.n_algebra = [0]
        self.group_params = 0
        self.algebra_params = 0
        for g in groups:
            self.n_group.append(g.group_params)
            self.n_algebra.append(g.algebra_params)
            self.group_params += g.group_params
            self.algebra_params += g.algebra_params

    def subgroup(self, a, i):
        start = 0
        for gi in range(i + 1):
            start += self.n_group[gi]
        end = start + self.n_group[i + 1]
        return a[start:end]

    def subalgebra(self, v, i):
        start = 0
        for gi in range(i + 1):
            start += self.n_algebra[gi]
        end = start + self.n_algebra[i + 1]
        return v[start:end]

    def product(self, a, b):
        assert a.shape[0] == self.group_params
        assert b.shape[0] == self.group_params
        return ca.vertcat(*[
            g.product(self.subgroup(a, i), self.subgroup(b, i))
            for i, g in enumerate(self.groups)])

    def inv(self, a):
        assert a.shape[0] == self.group_params
        return ca.vertcat(*[
            g.inv(self.subgroup(a, i))
            for i, g in enumerate(self.groups)])

    def exp(self, v):
        assert v.shape[0] == self.algebra_params
        return ca.vertcat(*[
            g.exp(self.subalgebra(v, i))
            for i, g in enumerate(self.groups)])

    def log(self, a):
        assert a.shape[0] == self.group_params
        return ca.vertcat(*[
            g.log(self.subgroup(a, i))
            for i, g in enumerate(self.groups)])