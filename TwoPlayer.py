import argparse
import torch as th
from torch import nn
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
        U = th.tensor([[0.8575, 0.3555, 0.2786, 0.5535, 0.9086, 0.7068, 0.7101, 0.5400],
        [0.0958, 0.5316, 0.8962, 0.3063, 0.0325, 0.3228, 0.5285, 0.5426],
        [0.6931, 0.0569, 0.6356, 0.3885, 0.4306, 0.0414, 0.9623, 0.4435],
        [0.7126, 0.4372, 0.0175, 0.6834, 0.5299, 0.9062, 0.5966, 0.9804],
        [0.4705, 0.9389, 0.0960, 0.4025, 0.5983, 0.5776, 0.1185, 0.7371],
        [0.2592, 0.6917, 0.5833, 0.1549, 0.0714, 0.5905, 0.8266, 0.7316],
        [0.3901, 0.3110, 0.0619, 0.6810, 0.8827, 0.4892, 0.6790, 0.1018],
        [0.1740, 0.1189, 0.4828, 0.2102, 0.4653, 0.1520, 0.2742, 0.8186]], requires_grad=False)
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
    Vf2 = th.zeros(K,1)
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
            Vf2[k] = th.mm( U, qfv[0:K,k:k+1] )[k]
        

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
            print( "d(pt,pavg):", kld( pt, ptavg ) )
            print( "d(p*,pavg):", kld( a, ptavg ) )
            print( "q.L:", q.L.t() )
            print( "p.L:", p.L.t() )
            print( "="*50 ) 

        QL = th.cat( ( QL, q.L.clone() ), dim = 1 )

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
