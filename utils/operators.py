import numpy as np
from numpy import ndarray, asarray
from typing import Optional, List, Union, Tuple

from utils.dataset_utils import sort_genes

class Operator():
    double: bool
    rvis: bool

    def __init__(self, function):
        self.function = function
        if self.rvis and self.function.__name__ != "rvis_concatenate":
            self.__name__ = "rvis_"+self.function.__name__
        else:
            self.__name__ = self.function.__name__

    def compute(self,x: Optional[Union[ndarray, Tuple[str, ndarray]]],
                y: Optional[Union[ndarray, Tuple[str, ndarray]]]) -> List[Optional[ndarray]]:

        if x is None or y is None:
            if self.double:
                return [None,None]
            else:
                return [None]
        else:
            return self.function(x,y)

class ConcatenateOperator(Operator):
    def __init__(self):
        self.double = False
        self.rvis = False
        super().__init__(concatenate)

class RVISConcatenateOperator(Operator):
    def __init__(self):
        self.double = False
        self.rvis = True
        super().__init__(rvis_concatenate)


class AverageOperator(Operator):
    def __init__(self):
        self.double = False
        self.rvis = False
        super().__init__(average)

class HadamardOperator(Operator):
    def __init__(self):
        self.double = False
        self.rvis = False
        super().__init__(hadamard)

class WeightedL1Operator(Operator):
    def __init__(self):
        self.double = False
        self.rvis = False
        super().__init__(weightedL1)

class WeightedL2Operator(Operator):
    def __init__(self):
        self.double = False
        self.rvis = False
        super().__init__(weightedL2)

class FirstOperator(Operator):
    def __init__(self):
        self.double = False
        self.rvis = False
        super().__init__(first)

class RVISFirstOperator(Operator):
    def __init__(self):
        self.double = False
        self.rvis = True
        super().__init__(first)

class SecondOperator(Operator):
    def __init__(self):
        self.double = False
        self.rvis = False
        super().__init__(second)

class RVISSecondOperator(Operator):
    def __init__(self):
        self.double = False
        self.rvis = True
        super().__init__(second)

class IdentityOperator(Operator):
    def __init__(self):
        self.double = False
        self.rvis = False
        super().__init__(identity)

def voidfunction(x: ndarray, y:ndarray):
    pass

def concatenate(x: Union[ndarray, Tuple[str,ndarray]],y: Union[ndarray, Tuple[str,ndarray]]):
    if type(x) is tuple:
        x = x[1]
        y = y[1]
    if x.ndim == 2:
        x_vec = np.concatenate([x[:,0],x[:,1],x[:,2],x[:,3]])
        y_vec = np.concatenate([y[:,0],y[:,1],y[:,2],y[:,3]])
        return [np.concatenate([x_vec,y_vec])]
    elif np.iscomplex(x[0]):
        return [np.concatenate([x.real, x.imag, y.real, y.imag])]
    else:
        return [np.concatenate([x,y])]

def rvis_concatenate(x: Union[ndarray, Tuple[str,ndarray]],y: Union[ndarray, Tuple[str,ndarray]]):
    gene_dict = {x[0]: x[1], y[0]: y[1]}
    sorted_genes = sort_genes([[x[0],y[0]]])
    x = gene_dict[sorted_genes[0][0]]
    y = gene_dict[sorted_genes[0][1]]

    if x.ndim == 2:
        x_vec = np.concatenate([x[:,0],x[:,1],x[:,2],x[:,3]])
        y_vec = np.concatenate([y[:,0],y[:,1],y[:,2],y[:,3]])
        return [np.concatenate([x_vec,y_vec])]
    elif np.iscomplex(x[0]):
        return [np.concatenate([x.real, x.imag, y.real, y.imag])]
    else:
        return [np.concatenate([x,y])]

def concatenate(x: Union[ndarray, Tuple[str,ndarray]],y: Union[ndarray, Tuple[str,ndarray]]):
    if type(x) is tuple:
        x = x[1]
        y = y[1]
    if x.ndim == 2:
        x_vec = np.concatenate([x[:,0],x[:,1],x[:,2],x[:,3]])
        y_vec = np.concatenate([y[:,0],y[:,1],y[:,2],y[:,3]])
        return [np.concatenate([x_vec,y_vec])]
    elif np.iscomplex(x[0]):
        return [np.concatenate([x.real, x.imag, y.real, y.imag])]
    else:
        return [np.concatenate([x,y])]

def average(x: Union[ndarray, Tuple[str,ndarray]],y: Union[ndarray, Tuple[str,ndarray]]):
    if type(x) is tuple:
        x = x[1]
        y = y[1]
    if x.ndim == 2:
        x_vec = np.concatenate([x[:,0],x[:,1],x[:,2],x[:,3]])
        y_vec = np.concatenate([y[:,0],y[:,1],y[:,2],y[:,3]])
        return [(x_vec+y_vec)/2]
    elif np.iscomplex(x[0]):
        return [np.concatenate([(x.real+y.real)/2, (x.imag+y.imag)/2])]
    else:
        return [(x+y)/2]

def hadamard(x: Union[ndarray, Tuple[str,ndarray]],y: Union[ndarray, Tuple[str,ndarray]]):
    if type(x) is tuple:
        x = x[1]
        y = y[1]
    if x.ndim == 2:
        x_vec = np.concatenate([x[:, 0], x[:, 1], x[:, 2], x[:, 3]])
        y_vec = np.concatenate([y[:, 0], y[:, 1], y[:, 2], y[:, 3]])
        return [x_vec*y_vec]
    elif np.iscomplex(x[0]):
        return [np.concatenate([(x*y).real,(x*y).imag])]
    else:
        return [x*y]

def weightedL1(x: Union[ndarray, Tuple[str,ndarray]],y: Union[ndarray, Tuple[str,ndarray]]):
    if type(x) is tuple:
        x = x[1]
        y = y[1]
    if x.ndim == 2:
        x_vec = np.concatenate([x[:, 0], x[:, 1], x[:, 2], x[:, 3]])
        y_vec = np.concatenate([y[:, 0], y[:, 1], y[:, 2], y[:, 3]])
        return [abs(x_vec-y_vec)]
    elif np.iscomplex(x[0]):
        return [np.concatenate([abs((x-y).real),abs((x-y).imag)])]
    else:
        return [abs(x-y)]

def weightedL2(x: Union[ndarray, Tuple[str,ndarray]],y: Union[ndarray, Tuple[str,ndarray]]):
    if type(x) is tuple:
        x = x[1]
        y = y[1]
    if x.ndim == 2:
        x_vec = np.concatenate([x[:, 0], x[:, 1], x[:, 2], x[:, 3]])
        y_vec = np.concatenate([y[:, 0], y[:, 1], y[:, 2], y[:, 3]])
        return [np.square(x_vec-y_vec)]
    elif np.iscomplex(x[0]):
        return [np.concatenate([np.square((x-y).real),np.square((x-y).imag)])]
    else:
        return [np.square(x-y)]

def first(x: Union[ndarray, Tuple[str,ndarray]],y: Union[ndarray, Tuple[str,ndarray]]):
    if type(x) is tuple:
        x = x[1]
        y = y[1]
    if x.ndim == 2:
        x_vec = np.concatenate([x[:, 0], x[:, 1], x[:, 2], x[:, 3]])
        return [x_vec]
    elif np.iscomplex(x[0]):
        return [np.concatenate([x.real,x.imag])]
    else:
        return [x]

def second(x: Union[ndarray, Tuple[str,ndarray]],y: Union[ndarray, Tuple[str,ndarray]]):
    if type(x) is tuple:
        x = x[1]
        y = y[1]
    if x.ndim == 2:
        y_vec = np.concatenate([y[:, 0], y[:, 1], y[:, 2], y[:, 3]])
        return [y_vec]
    elif np.iscomplex(x[0]):
        return [np.concatenate([y.real,y.imag])]
    else:
        return [y]

def identity(x: Union[ndarray, Tuple[str,ndarray]],y: Union[ndarray, Tuple[str,ndarray]]):
    if type(x) is tuple:
        x = x[1]
        y = y[1]
    return [[x,y]]

map_name_to_transform = {
    "average": AverageOperator(),
    "hadamard": HadamardOperator(),
    "weightedl1": WeightedL1Operator(),
    "weightedl2": WeightedL2Operator(),
    "rvis-concatenate": RVISConcatenateOperator(),
    "rand-concatenate": ConcatenateOperator()
}
