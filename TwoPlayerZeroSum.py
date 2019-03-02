import torch as th
import torch.optim
from torch import nn
from torch.distributions.categorical import Categorical
from math import log
from player import *


T = 500000
optimistic = False

th.set_default_tensor_type(th.DoubleTensor)
#th.manual_seed(3)


U = th.rand( 4, 4, requires_grad=False )
U = U.type(th.DoubleTensor)

K = U.size(0)

eta = (log(K)/T)**0.25 if optimistic else (log(K)/T)**0.5

print(U)

p = player(K = K, eta = eta, optimistic = optimistic)
q = player(K = K, eta = eta, optimistic = False)

Pt = th.zeros( K, 1, requires_grad=False )

Vavg = 0
ptavg = th.zeros(K,1)
qtavg = th.zeros(K,1)
Vf = th.zeros(K,1)
qfv = th.zeros(K,K)

stint = 3


for i in range(1,T+1):
    pt = p.play()
    qt = q.play()

    for k in range(K):
        if i > 1:
            qfv[0:K,k:k+1] = q.policyplay(k)
        else:
            qfv[0:K,k:k+1] = qt
        Vf[k] = (Vf[k]/i)*(i-1) + th.mm( U, qfv[0:K,k:k+1] )[k]/i

    q.loss( -th.mm(pt.t(),U).t() )
    p.loss( th.mm(U, qt) )

    V = th.mm( pt.t(), th.mm(U, qt) )
     
    Vavg = (Vavg/i)*(i-1) + V/i
    ptavg = (ptavg/i)*(i-1) + pt/i
    qtavg = (qtavg/i)*(i-1) + qt/i

    Pt = th.cat( (Pt, p.L), dim = 1 )

    if log( i ) > stint:
        stint += 0.25
        print("Round:",i)
        #print( max(th.mm(ptavg.t(),U).t()), min(th.mm(U, qtavg)) )
        print( log( Vavg - min(Vf) ) )
        print( log( Vavg - min(th.mm(U, qtavg)) ) )
        print( th.argmin(th.mm(U, qtavg)) )
        #print(pt, qt)

sigma = torch.nn.Softmax(dim=0)
ar = th.randn(K,1, requires_grad=True)
optimizer = torch.optim.SGD( [ar], lr = 0.001)

for i in range(1000):
    optimizer.zero_grad()
    a = sigma(ar)
    Pat = Pt + a
    Qt = th.mm( U, sigma(-eta * Pat ) )
    ut = th.mm( a.t(), Qt )
    R = th.sum(ut)
    R.backward()
    optimizer.step()

print( "a*:", sigma(ar).t() )

print("value:", Vavg) #0.08348
print("loss of q:", th.mm(ptavg.t(),U)) #(0.2183,0.7817)
print("qt:", qtavg.t() )
print("loss of p:", th.mm(U, qtavg).t()) #(0.7187,0.2813)
print("pt:", ptavg.t() )
