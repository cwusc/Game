import torch as th
from torch import nn
from torch.distributions.categorical import Categorical
from math import log
from player import *


T = 1000000
optimistic = True

th.set_default_tensor_type(th.DoubleTensor)
#th.manual_seed(3)

#A = th.tensor([[0.7,0.2],[0.2,0.5]])
A = th.rand(8,8)
A = A.type(th.DoubleTensor)

K = A.size(0)

eta = (log(K)/T)**0.25 if optimistic else (log(K)/T)**0.5

print(A)

p = player(K = K, eta = eta, optimistic = optimistic)
q = player(K = K, eta = eta, optimistic = optimistic)

Vavg = 0
ptavg = th.zeros(K,1)
qtavg = th.zeros(K,1)

stint = 7

for i in range(1,T+1):
    pt = p.play()
    qt = q.play()
    q.loss( -th.mm(pt.t(),A).t() )
    p.loss( th.mm(A, qt) )

    V = th.mm( pt.t(), th.mm(A, qt) )
     
    Vavg = (Vavg/i)*(i-1) + V/i
    ptavg = (ptavg/i)*(i-1) + pt/i
    qtavg = (qtavg/i)*(i-1) + qt/i
    if log( i ) > stint:
        stint += 0.25
        print( max(th.mm(ptavg.t(),A).t()), min(th.mm(A, qtavg)), 
               log( Vavg - min(th.mm(A, qtavg)) ) )
        #print(pt, qt)


print("value:", Vavg) #0.08348
print("loss of q:", th.mm(ptavg.t(),A)) #(0.2183,0.7817)
print("qt:", qtavg.t() )
print("loss of p:", th.mm(A, qtavg).t()) #(0.7187,0.2813)
print("pt:", ptavg.t() )
