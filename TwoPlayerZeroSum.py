import torch as th
from torch import nn
from torch.distributions.categorical import Categorical
from math import log

class player:
    def __init__( self, eta = 1, K = 1 ):
        self.eta = eta
        self.L = th.zeros(K,1)
    def play( self ):
        return nn.Softmax(dim = 0)( -self.eta * self.L )
    def loss( self, lt ):
        self.L += lt

K = 2
T = 1000000

th.manual_seed(3)

eta = (log(K)/T)**0.5
A = th.rand(K,K)
print(A)
p = player(K = K, eta = eta)
q = player(K = K, eta = eta)

for i in range(T):
    pt = p.play()
    qt = q.play()
    q.loss( -th.mm(pt.t(),A).t() )
    p.loss( th.mm(A, qt) )
    if i% int(T/10) == 0:
        print( th.mm( pt.t(), th.mm(A, qt) ) )

print(th.mm(pt.t(),A))
print(th.mm(A, qt).t())
print(pt.t())
print(qt.t())