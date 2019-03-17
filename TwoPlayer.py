import argparse
import torch as th
from util import *
from player import *
from model import *

def run( args, G = None, Gp = None, Gq = None ):
    optimistic = (args.optimistic==1)
    zerosum = (args.ZeroSum == 1)
    nrand = args.nrand

    th.set_default_tensor_type(th.DoubleTensor)

    if nrand <= 0:
        #C = th.tensor([[0.6927, 0.0188],[0.3149, 0.7550]], requires_grad=False)
        Cp = th.tensor([[0.5,0.5],[0.51,0]], requires_grad=False) #Stop & Go
        Cq = th.tensor([[0.5,0.51],[0.5,0]], requires_grad=False) #Stop & Go
    elif G is None:
        Cp = th.rand( nrand, nrand )
        Cq = th.rand( nrand, nrand )
    elif zerosum:
        Cp = G
    else:
        Cp,Cq = Gp,Gq

    K = Cp.size(0)
    T = args.T

    if args.lr > 0:
        eta = args.lr
    else:
        (log(K)/T)**0.25 if optimistic else (log(K)/T)**0.5

    print("="*50)
    if zerosum:
        Cq = -Cp.t()
        print(Cp)
    else:
        print("p Loss:\n", Cp)
        print("q Loss:\n", Cq)
    print("="*50)

    torch.set_printoptions(precision=3, threshold=200, edgeitems=2, linewidth=180, profile=None)

    p = player(K = K, eta = eta, optimistic = optimistic)
    q = player(K = K, eta = eta, optimistic = optimistic)

    Vavg = 0
    ptavg = th.zeros(K,1)
    qtavg = th.zeros(K,1)
    Vf = th.zeros(K,1)
    QL = q.L.clone()

    stint = 7

    for tt in range(1,T+1):
        pt = p.play()
        qt = q.play()

        q.loss( th.mm(Cq, pt) )
        p.loss( th.mm(Cp, qt) )

        V = th.mm( pt.t(), th.mm(Cp, qt) )
        Vavg = (Vavg/tt)*(tt-1) + V/tt
        ptavg = (ptavg/tt)*(tt-1) + pt/tt
        qtavg = (qtavg/tt)*(tt-1) + qt/tt

        if tt == T:
            break
        
        for k in range(K):
            pk = th.tensor( [[ float(j==k) for j in range(K)]] ).t()
            Vf[k] += th.mm( Cp, q.policyplay( th.mm(Cq, pk) ) )[k]

        if log( tt ) > stint:
            stint += 0.2
            Pa = min(Vf)/tt
            Pp, a = solve(QL =QL, e = q.eta, K = K, Cq = Cq, Cp = Cp, o = q.optmstc, l = Pa )
            print("Round:",tt)
            print( "Policy Regret a*:", mylog( Vavg - Pa ) )
            print( "Policy Regret p*:", mylog( Vavg - Pp ) )
            print( " External Regret:", mylog( Vavg - min(th.mm(Cp, qtavg))) )
            print( "  a*:", int(th.argmin(Vf)) )
            print( "  p*:", a.t() )
            print( "pavg:", ptavg.t() )
            print( "  pt:", pt.t() )
            print( "d(pt,pavg):", kld( pt, ptavg ) )
            print( "d(p*,pavg):", kld( a, ptavg ) )
            print( "  qt:", pt.t() )
            print( " q.L:", q.L.t() )
            print( " p.L:", p.L.t() )
            print("loss of p:", th.mm(Cp,qtavg).t()) 
            print("loss of q:", th.mm(Cq,ptavg).t())
            print( "="*50 ) 

        QL = th.cat( ( QL, q.L.clone() ), dim = 1 )

    return True

def LRtuning(args):
    G = th.rand( args.nrand, args.nrand )
    print(G)
    for i in range(1,-9,-1):
        args.lr = 2.0**i
        print(run(args, G=G), 2.0**i)

parser = argparse.ArgumentParser(description='Set variant algos')

parser.add_argument('--opt', type=int, default=1, dest = 'optimistic')
parser.add_argument('--n', type=int, default=0, dest='nrand')
parser.add_argument('--lr', type=float, default=0.5, dest='lr')
parser.add_argument('--t', type=int, default=100000, dest='T')
parser.add_argument('--z', type=int, default=1, dest='ZeroSum')

args = parser.parse_args()

run(args)
