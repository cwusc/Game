import torch as th
from torch import nn
from torch.distributions.categorical import Categorical
from math import log
from util import *

th.set_default_tensor_type(th.DoubleTensor)

class metaplayer:
    def __init__( self, eta = 1, K = 1, optimistic = False, bandits = False):
        self.eta = eta
        self.bandits = bandits
        self.K = K
        self.plyrs = []
        for i in range(K):
            p = player( K = K, eta = eta, optimistic = optimistic, bandits = bandits )
            self.plyrs.append( p )
        self.pt = None
    def play( self, eta = -1 ):
        for i in range(self.K):
            pi = self.plyrs[i].play( eta )
            if i == 0:
                self.Q = pi 
            else:
                self.Q = th.cat( (self.Q, pi), dim = 1 )
        if self.pt is None:
            self.pt = th.ones( self.K, 1 ) / self.K
        while True:
            old = self.pt
            self.pt = th.mm( self.Q, self.pt )
            if th.norm( self.pt-old ) < 1e-7:
                return self.pt
    def loss( self, lossvec ):
        for i in range(self.K):
            self.plyrs[i].loss( self.pt[i] * lossvec )


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
    
    def loss( self, lossvec ):
        if self.bandits:
            lte = lossvec[ self.a ] / self.pt[ self.a ]
            lossvec = lte * onehot( self.a, self.K )
        self.L += lossvec
        self.last = lossvec
