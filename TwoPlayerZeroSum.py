import argparse
import torch as th
from torch import nn
from torch.distributions.categorical import Categorical
from math import log
from util import *
from player import *
from model import *

parser = argparse.ArgumentParser(description='Set variant algos')

T = 100000
parser.add_argument('--opt', type=int, default=1, dest = 'optimistic')
parser.add_argument('--n', type=int, default=0, dest='nrand')

args = parser.parse_args()

optimistic = (args.optimistic==1)
nrand = args.nrand

th.set_default_tensor_type(th.DoubleTensor)

if nrand <= 0:
    #U = th.tensor([[0,1,-1],[-1,0,1],[1,-1,0]], requires_grad=False)
    U = th.tensor([[0.6927, 0.0188],[0.3149, 0.7550]], requires_grad=False)
else:
    U = th.rand( nrand, nrand )

U = U.type(th.DoubleTensor)
K = U.size(0)
eta = 0.5 #(log(K)/T)**0.25 if optimistic else (log(K)/T)**0.5

print(U)

p = player(K = K, eta = eta, optimistic = optimistic)
q = player(K = K, eta = eta, optimistic = optimistic)
#q = player(K = K, eta = (log(K)/T)**0.5, optimistic = False)

Vavg = 0
ptavg = th.zeros(K,1)
qtavg = th.zeros(K,1)
Vf = th.zeros(K,1)
qfv = th.zeros(K,K)

QL = q.L.clone()

stint = 3


for tt in range(1,T+1):
    pt = p.play()
    qt = q.play()

    q.loss( -th.mm(pt.t(),U).t() )
    p.loss( th.mm(U, qt) )

    V = th.mm( pt.t(), th.mm(U, qt) )
    Vavg = (Vavg/tt)*(tt-1) + V/tt
    ptavg = (ptavg/tt)*(tt-1) + pt/tt
    qtavg = (qtavg/tt)*(tt-1) + qt/tt

    if tt == T:
        break

    for k in range(K):
        pk = th.tensor( [[ float(j==k) for j in range(K)]] )
        Lk = -th.mm( pk, U ).t()
        qfv[0:K,k:k+1] = q.policyplay(Lk)
        Vf[k] = ( Vf[k]*(tt-1) + th.mm( U, qfv[0:K,k:k+1] )[k] ) / (tt)

    if log( tt ) > stint:
        stint += 0.25
        R, a = solve(QL =QL, e = q.eta, K = K, U = U, o = q.optmstc)
        print("Round:",tt)
        print( "Policy Regret a*:", mylog( Vavg - min(Vf) ) )
        print( "Policy Regret p*:", mylog( Vavg - R/tt ) )
        print( " External Regret:", mylog( Vavg - min(th.mm(U, qtavg))) )
        print( "a*:", int(th.argmin(th.mm(U, Vf))) )
        print( "p*:", a.t() )
        print( "pt:", ptavg.t() )
        print( "="*50 ) 

    QL = th.cat( (QL, q.L), dim = 1 )

R, a = solve(QL =QL, e = q.eta, K = K, U = U, o = q.optmstc)
print( "final p* V:", R/(tt) )
print( "final p*:", a.t() )
print( "final Vf:", Vf.t() )
print( "final a*:", th.argmin(Vf) )

print("value:", Vavg) 
print("loss of q:", th.mm(ptavg.t(),U)) 
print("qt:", qtavg.t() )
print("loss of p:", th.mm(U, qtavg).t()) 
print("pt:", ptavg.t() )
