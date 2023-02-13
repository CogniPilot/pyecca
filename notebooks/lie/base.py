import casadi as ca
import abc


class LieGroup(abc.ABC):
    """
    A Lie Group with group operator (*) is:
    
    (C)losed under operator (*)
    (A)ssociative with operator (*), (G1*G2)*G3 = G1*(G2*G3)
    (I)nverse: has an inverse such that G*G^-1 = e
    (N)uetral: has a neutral element: G*e = G
    """
    
    def __init__(self, param):
        self.param = ca.SX(param)
    
    def __mul__(self, other):
        """
        The * operator will be used as the Group multiplication operator
        (see product)
        """
        if type(self) != type(other):
            return TypeError("Lie Group types must match for product")
        assert isinstance(other, LieGroup)
        return self.product(other)

    @abc.abstractmethod
    def identity(self):
        """
        The identity element of the gorup, e
        """
        ...
 
    @abc.abstractmethod
    def product(self, other):
        """
        The group operator (*), returns an element of the group: G1*G2 = G3
        """
        ...
    
    @abc.abstractmethod
    def inv(self):
        """
        The inverse operator G1*G1.inv() = e
        """
        ...

    @abc.abstractmethod
    def log(self):
        """
        Returns the Lie logarithm of a group element, an element of the
        Lie algebra
        """
        ...


    @abc.abstractmethod
    def to_matrix_lie_group(self):
        """
        Returns the matrix lie group representation
        """
        ...
    
    def __repr__(self):
        return repr(self.param)

    def __str__(self):
        return str(self.param)
    
    def __eq__(self, other) -> bool:
        return self.param == other.param


class LieAlgebra(abc.ABC):

    def __init__(self, param):
        self.param = ca.SX(param)
        
    def __add__(self, other):
        return self.add(other)

    def __sub__(self, other):
        return self.sub(other.neg())

    def __rmul__(self, other):
        return self.rmul(other)

    def __neg__(self):
        return self.neg()
    
    def __eq__(self, other) -> bool:
        return self.param == other.param

    @abc.abstractmethod
    def neg(self, other):
        """
        Negative of Lie algebra
        """
        ...

    @abc.abstractmethod
    def add(self, other):
        """
        Add to elements of the Lie algebra
        """
        ...

    @abc.abstractmethod
    def rmul(self, other):
        """
        Add to elements of the Lie algebra
        """
        ...

    @abc.abstractmethod
    def wedge(self, other):
        ...
    
    @abc.abstractmethod
    def vee(self):
        ...

    @abc.abstractmethod
    def exp(self):
        ...

    def __repr__(self):
        return repr(self.param)

    def __str__(self):
        return str(self.param)