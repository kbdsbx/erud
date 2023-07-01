# update parameters functions

from erud.upf.norm import norm as norm_class
from erud.upf.momentum import momentum as momentum_class
from erud.upf.adam import adam as adam_class

def norm (*args) :
    return norm_class(*args).updateFunc

def momentum (*args) :
    return momentum_class(*args).updateFunc

def adam (*args) :
    return adam_class(*args).updateFunc