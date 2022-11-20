import sympy
import numpy
import casadi as ca

class SympyToCasadi:
    
    def __init__(self):
        self.symbols = {}

    def parse(self, f, f_dict=[] ,depth=0, cse=False, verbose=False):
        prs = lambda f: self.parse(f, f_dict, depth=depth+1, cse=False, verbose=verbose)
        f_type = type(f)
        dict_keys = list(f_dict.keys())
        if verbose:
            print('-'*depth, f, 'type', f_type)
        if cse:
            cse_defs, cse_exprs = sympy.cse(f)
            assert len(cse_exprs) == 1
            ca_cse_defs = {}
            for symbol, subexpr in reversed(cse_defs):
                ca_cse_defs[prs(symbol)] = prs(subexpr)
            f_ca = prs(cse_exprs[0])
            for k, v in ca_cse_defs.items():
                f_ca = ca.substitute(f_ca, k, v)
            for symbol, subexpr in reversed(cse_defs):
                if str(symbol) in self.symbols:
                    self.symbols.pop(str(symbol))
            return f_ca
        if f_type == sympy.core.add.Add:
            s = 0
            for arg in f.args:
                s += prs(arg)
            return s
        elif f_type == sympy.core.mul.Mul:
            prod = 1
            for arg in f.args:
                prod *= prs(arg)
            return prod
        elif f_type == sympy.core.numbers.Integer:
            return int(f)
        elif f_type == sympy.core.power.Pow:
            base, power = f.args
            base_ca = prs(base)
            if type(power) == sympy.core.numbers.Half:
                return ca.sqrt(base_ca)
            else:
                return base_ca**prs(power)
        elif f_type == sympy.core.symbol.Symbol:
            if str(f) not in self.symbols:
                self.symbols[str(f)] = ca.SX.sym(str(f))
            return self.symbols[str(f)]
        elif f_type == sympy.matrices.dense.MutableDenseMatrix:
            mat = ca.SX(f.shape[0], f.shape[1])
            for i in range(f.shape[0]):
                for  j in range(f.shape[1]):                    
                    mat[i, j] = prs(f[i, j])
            return mat
        elif f_type == sympy.core.numbers.Float: #Convert Float to int
            return int(f)
        elif f_type == sympy.core.numbers.One:
            return 1
        elif f_type == sympy.core.numbers.Zero:
            return 0
        elif f_type == sympy.core.numbers.NegativeOne:
            return -1
        elif f_type == sympy.core.numbers.Half:
            return 0.5
        elif str(f_type) == 'sin':
            return ca.sin(prs(f.args[0]))
        elif str(f_type) == 'cos':
            return ca.cos(prs(f.args[0]))
        elif str(f_type) in dict_keys:
            for i in range(len(dict_keys)):
                return f_dict[dict_keys[i]](prs(f.args[0]))
        else:
            print('unhandled type', type(f), f)