import sys
import numpy as np

sys.path.append("../")
import pytest as test
from core.erud.graph import graph
from core.erud.basic_operators.variable import variable

def test_init () :
    a = variable(np.array([1,2,3]))
    b = variable(np.array([2,3,4]))
    c = variable(np.array([3,4,5]))

    d = a + b + c

    print(d.forward_value())

