import torch as th
from torch import nn
from torch.distributions.categorical import Categorical
from math import log
from util import *

th.set_default_tensor_type(th.DoubleTensor)

class player:
    def __init__( self, eta = 1, K = 1, optimistic = False, bandits = False):
        self.eta = eta
        self.L = th.zeros(K,1)
        self.last = th.zeros(K,1)
        self.optmstc = optimistic
        self.bandits = bandits
        self.K = K

    def play( self, eta = -1 ):
        if eta >= 0:
            self.eta = eta
        if self.optmstc:
            L = self.L + self.last
        else:
            L = self.L
        self.pt = nn.Softmax(dim = 0)( -self.eta * ( L ) )
        self.a = sample( self.pt )
        if not self.bandits: 
            return self.pt 
        return onehot( self.a, self.K )

    def policyplay( self, La, eta = -1 ):
        if eta >= 0:
            self.eta = eta
        if self.optmstc:
            L = self.L - self.last + La * 2
        else:
            L = self.L - self.last + La
        return nn.Softmax(dim = 0)( -self.eta * ( L ) )
    
    def loss( self, lt ):
        if self.bandits:
            lt = lt / self.pt[ self.a ]
        self.L += lt
        self.last = lt 
