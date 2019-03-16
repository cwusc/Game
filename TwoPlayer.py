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
        Up = th.tensor([[0.5,0.5],[0.51,0]], requires_grad=False) #Stop & Go
        Uq = th.tensor([[0.5,0.51],[0.5,0]], requires_grad=False) #Stop & Go
    elif G is None:
        Up = th.rand( nrand, nrand )
        Uq = th.rand( nrand, nrand )
    else:
        Up,Uq = G

    Up = Up.type(th.DoubleTensor)
    Uq = Uq.type(th.DoubleTensor)
    K = Up.size(0)


    T = args.T

    if args.lr > 0:
        eta = args.lr
    else:
        (log(K)/T)**0.25 if optimistic else (log(K)/T)**0.5
    #print("p Util:")
    #print(Up)
    #print("q Util:")
    #print(Uq)
    #print("="*50)
    torch.set_printoptions(precision=3, threshold=200, edgeitems=2, linewidth=120, profile=None)

    p = player(K = K, eta = eta, optimistic = optimistic)
    q = player(K = K, eta = eta, optimistic = optimistic)

    Vavg = 0
    ptavg = th.zeros(K,1)
    qtavg = th.zeros(K,1)
    Vf = th.zeros(K,1)
    Vf2 = th.zeros(K,1)
    qfv = th.zeros(K,K)

    #QL = q.L.clone()

    stint = 7

    for tt in range(1,T+1):
        pt = p.play()
        qt = q.play()

        q.loss( -th.mm(pt.t(),Uq).t() )
        p.loss( -th.mm(Up, qt) )

        V = th.mm( pt.t(), th.mm(Up, qt) )
        Vavg = (Vavg/tt)*(tt-1) + V/tt
        ptavg = (ptavg/tt)*(tt-1) + pt/tt
        qtavg = (qtavg/tt)*(tt-1) + qt/tt

        if tt == T:
            break

        for k in range(K):
            pk = th.tensor( [[ float(j==k) for j in range(K)]] )
            Lk = -th.mm( pk, Uq ).t()
            qfv[0:K,k:k+1] = q.policyplay(Lk)
            Vf[k] = ( Vf[k]*(tt-1) + th.mm( Up, qfv[0:K,k:k+1] )[k] ) / (tt)
            Vf2[k] = th.mm( Up, qfv[0:K,k:k+1] )[k]
        

        if log( tt ) > stint:
            if min(ptavg) < 0.01: 
                return False
            stint += 0.25
            #R, a = solve( QL = QL, e = q.eta, K = K, U = Uq, o = q.optmstc)
            print("Round:",tt)
            print( "Policy Regret a*:", mylog( -Vavg + max(Vf) ) )
            #print( "Policy Regret p*:", mylog( Vavg - R/tt ) )
            print( " External Regret:", mylog( -Vavg + max(th.mm(Up, qtavg))) )
            print( "  a*:", int(th.argmin(th.mm(Up, Vf))) )
            #print( "  p*:", a.t() )
            print( "pavg:", ptavg.t() )
            print( "  pt:", pt.t() )
            print( "d(pt,pavg):", kld( pt, ptavg ) )
            #print( "d(p*,pavg):", kld( a, ptavg ) )
            print( "q.L:", q.L.t() )
            print( "p.L:", p.L.t() )
            print( "="*50 ) 

        #QL = th.cat( ( QL, q.L.clone() ), dim = 1 )

    #R, a = solve(QL =QL, e = q.eta, K = K, U = U, o = q.optmstc)
    #print( "final p* V:", R/(tt) )
    #print( "final p*:", a.t() )
    print( "final Vf:", Vf.t() )
    print( "final a*:", th.argmin(Vf) )

    print("value:", Vavg) 
    print("loss of q:", th.mm(ptavg.t(),Uq)) 
    print("qt:", qtavg.t() )
    print("loss of p:", th.mm(Up, qtavg).t()) 
    print("pt:", ptavg.t() )
    print("="*50)
    print("p Util:")
    print(Up)
    print("q Util:")
    print(Uq)
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

c = 0
while not run(args):
    c += 1
    print(c)
