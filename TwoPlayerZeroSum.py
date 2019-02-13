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

Vsum = 0
ptsum = th.zeros(K,1)
qtsum = th.zeros(K,1)

for i in range(T):
    pt = p.play()
    qt = q.play()
    q.loss( -th.mm(pt.t(),A).t() )
    p.loss( th.mm(A, qt) )

    V = th.mm( pt.t(), th.mm(A, qt) )
    
    Vsum += V
    ptsum += pt
    qtsum += qt

Vsum/=T
qtsum/=T
ptsum/=T

print("value:", Vsum) #0.08348
print("loss of q:", th.mm(ptsum.t(),A)) #(0.2183,0.7817)
print("qt:", qtsum.t() )
print("loss of p:", th.mm(A, qtsum).t()) #(0.2813,0.7187)
print("pt:", ptsum.t() )
