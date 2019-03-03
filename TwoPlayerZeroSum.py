import argparse
import torch as th
import torch.optim
from torch import nn
from torch.distributions.categorical import Categorical
from math import log
from player import *
from model import *

parser = argparse.ArgumentParser(description='Set variant algos')

T = 10000
parser.add_argument('--opt', type=int, default=1, dest = 'optimistic')
parser.add_argument('--n', type=int, default=0, dest='nrand')

args = parser.parse_args()

optimistic = (args.optimistic==1)
nrand = args.nrand

th.set_default_tensor_type(th.DoubleTensor)

if nrand <= 0:
    U = th.tensor([[0,9,-2],[-9,0,1],[2,-1,0]], requires_grad=False)
    #U = th.tensor([[0,1,-1],[-1,0,1],[1,-1,0]], requires_grad=False)
    #U = th.tensor([[0.6310, 0.4909], [0.5340, 0.0578]], requires_grad=False)
else:
    U = th.rand( nrand, nrand )

U = U.type(th.DoubleTensor)
K = U.size(0)
eta = (log(K)/T)**0.25 if optimistic else (log(K)/T)**0.5

print(U)

p = player(K = K, eta = eta, optimistic = optimistic)
#q = player(K = K, eta = eta, optimistic = optimistic)
q = player(K = K, eta = (log(K)/T)**0.5, optimistic = False)

Vavg = 0
ptavg = th.zeros(K,1)
qtavg = th.zeros(K,1)
Vf = th.zeros(K,1)
qfv = th.zeros(K,K)

QL = q.L.clone()

stint = 3

for i in range(1,T+1):
    pt = p.play()
    qt = q.play()

    q.loss( -th.mm(pt.t(),U).t() )
    p.loss( th.mm(U, qt) )

    V = th.mm( pt.t(), th.mm(U, qt) )
    Vavg = (Vavg/i)*(i-1) + V/i
    ptavg = (ptavg/i)*(i-1) + pt/i
    qtavg = (qtavg/i)*(i-1) + qt/i

    if i == T:
        break

    QL = th.cat( (QL, q.L), dim = 1 )
    for k in range(K):
        pk = th.tensor( [[ float(j==k) for j in range(K)]] )
        Lk = -th.mm( pk, U ).t()
        qfv[0:K,k:k+1] = q.policyplay(Lk)
        Vf[k] = ( Vf[k]*(i-1) + th.mm( U, qfv[0:K,k:k+1] )[k] ) / (i)

    if log( i ) > stint:
        stint += 0.25
        print("Round:",i)
        print( "  Policy Regret:", log(abs(Vavg - min(Vf))) )
        print( "External Regret:", log(abs(Vavg - min(th.mm(U, qtavg)))) )
        print( "a*:", th.argmin(th.mm(U, Vf)) )


md = Model(eta = q.eta, K = K, U = U, optimistic = q.optmstc)
optimizer = torch.optim.SGD( [md.ar], lr = 0.01)
QL.requires_grad = False

for i in range(10000):
    optimizer.zero_grad()
    R = md( QL )
    R.backward()
    optimizer.step()

print( md( QL )/(T-1) )
print( "final p*:", md.sigma(md.ar).t() )
print( Vf.t() )
print( "final a*:", th.argmin(Vf) )

print("value:", Vavg) 
print("loss of q:", th.mm(ptavg.t(),U)) 
print("qt:", qtavg.t() )
print("loss of p:", th.mm(U, qtavg).t()) 
print("pt:", ptavg.t() )
