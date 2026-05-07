import numpy as np
from sympy import Matrix, pprint, Eq
from sympy.matrices.expressions.hadamard import HadamardProduct

A = Matrix([[1, 2], [3, 4]])
B = Matrix([[0.5, 0.5], [0.5, 0.5]])
res = Matrix([[0.5, 1], [1.5, 2]])

expr = HadamardProduct(A, B, evaluate=False)
pprint(Eq(expr, res, evaluate=False))
