import argparse
import torch as th
from torch import nn
from torch.distributions.categorical import Categorical
from math import log
from util import *
from player import *
from model import *

def run( args, G = None ):
    optimistic = (args.optimistic==1)
    nrand = args.nrand

    th.set_default_tensor_type(th.DoubleTensor)

    if nrand <= 0:
        #U = th.tensor([[0,1,-1],[-1,0,1],[1,-1,0]], requires_grad=False)
        #U = th.tensor([[0.6927, 0.0188],[0.3149, 0.7550]], requires_grad=False)
        U = th.tensor([[0.8864, 0.2726, 0.0597, 0.8586, 0.8989, 0.9384],
        [0.9910, 0.0314, 0.4127, 0.8870, 0.5217, 0.0868],
        [0.6146, 0.9306, 0.2217, 0.9167, 0.8181, 0.6778],
        [0.7955, 0.3742, 0.6322, 0.1813, 0.0647, 0.2982],
        [0.4699, 0.3273, 0.5085, 0.9462, 0.0165, 0.3451],
        [0.0110, 0.2700, 0.8516, 0.0158, 0.2393, 0.2470]], requires_grad=False)
    elif G is None:
        U = th.rand( nrand, nrand )
    else:
        U = G

    U = U.type(th.DoubleTensor)
    K = U.size(0)


    T = args.T

    if args.lr > 0:
        eta = args.lr
    else:
        (log(K)/T)**0.25 if optimistic else (log(K)/T)**0.5

    print(U)
    torch.set_printoptions(precision=3, threshold=200, edgeitems=2, linewidth=120, profile=None)

    p = player(K = K, eta = eta, optimistic = optimistic)
    q = player(K = K, eta = eta, optimistic = optimistic)

    Vavg = 0
    ptavg = th.zeros(K,1)
    qtavg = th.zeros(K,1)
    Vf = th.zeros(K,1)
    qfv = th.zeros(K,K)

    QL = q.L.clone()

    stint = 5

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
            R, a = solve( QL = QL, e = q.eta, K = K, U = U, o = q.optmstc)
            print("Round:",tt)
            print( "Policy Regret a*:", mylog( Vavg - min(Vf) ) )
            print( "Policy Regret p*:", mylog( Vavg - R/tt ) )
            print( " External Regret:", mylog( Vavg - min(th.mm(U, qtavg))) )
            print( "  a*:", int(th.argmin(th.mm(U, Vf))) )
            print( "  p*:", a.t() )
            print( "pavg:", ptavg.t() )
            print( "  pt:", pt.t() )
            print( "d(pt,pavg):", th.norm( pt - ptavg ) )
            print( "d(p*,pavg):", th.norm( a - ptavg ) )
            print( "q.L:", q.L.t() )
            print( "p.L:", p.L.t() )
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
    return True

def LRtuning(args):
    G = th.rand( args.nrand, args.nrand )
    print(G)
    for i in range(1,-9,-1):
        args.lr = 2.0**i
        print(run(args, G), 2.0**i)

parser = argparse.ArgumentParser(description='Set variant algos')

parser.add_argument('--opt', type=int, default=1, dest = 'optimistic')
parser.add_argument('--n', type=int, default=0, dest='nrand')
parser.add_argument('--lr', type=float, default=0.5, dest='lr')
parser.add_argument('--t', type=int, default=100000, dest='T')

args = parser.parse_args()

run(args)
