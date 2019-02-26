import torch as th
from torch import nn
from torch.distributions.categorical import Categorical
from math import log

class player:
    def __init__( self, eta = 1, K = 1, optimistic = False ):
        self.eta = eta
        self.L = th.zeros(K,1)
        self.last = th.zeros(K,1)
        self.optmstc = optimistic

    def play( self, eta = -1 ):
        if eta >= 0:
            self.eta = eta
        return nn.Softmax(dim = 0)( -self.eta * ( self.L + self.last ) )

    def policyplay( self, a, eta = -1 ):
        if eta >= 0:
            self.eta = eta
        onehot = th.zero(self.last)
        onehot[ a ] = 2 if self.optmstc else 1
        return nn.Softmax(dim = 0)( -self.eta * ( self.L - self.last + onehot ) )
    
    def loss( self, lt ):
        self.L += lt
        self.last = lt if self.optmstc else 0
