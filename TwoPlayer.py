import argparse
from util import *
from player import *
from model import *

def run( args, G = None, Gp = None, Gq = None, mix = False):
    optimistic = (args.optimistic==1)
    nrand = args.nrand
    th.set_default_tensor_type(th.DoubleTensor)

    if nrand <= 0:
        #Cp = th.tensor([[0,1,-1],[-1,0,1],[1,-1,0]], requires_grad=False)
        #C = th.tensor([[0.6927, 0.0188],[0.3149, 0.7550]], requires_grad=False)
        Cp = th.tensor([[0.5,0.5],[0.51,0]], requires_grad=False) #Stop & Go
        Cq = th.tensor([[0.5,0.51],[0.5,0]], requires_grad=False) #Stop & Go
    elif G is not None:
        Cp = G
    elif Gp is not None:
        Cp,Cq = Gp,Gq
    else:
        Cp = th.rand( nrand, nrand )
        Cq = th.rand( nrand, nrand )

    if args.ZeroSum == 1:
        Cq = -Cp.t()

    K = Cp.size(0)
    T = args.T

    if args.lr > 0:
        eta = args.lr
    else:
        (log(K)/T)**0.25 if optimistic else (log(K)/T)**0.5

    print("="*50)
    print("p Loss:\n", Cp)
    print("q Loss:\n", Cq)
    print("="*50)

    #torch.set_printoptions(precision=3, threshold=200, edgeitems=2, linewidth=180, profile=None)

    p = player(K = K, eta = eta, optimistic = optimistic)
    q = player(K = K, eta = eta, optimistic = optimistic)

    Vavg = 0
    ptavg = th.zeros(K,1)
    qtavg = th.zeros(K,1)
    Vf = th.zeros(K,1)
    QL = q.L.clone()
    stint = 5

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
            if mix and min(ptavg) < 0.01:
                return False
            stint += 0.2
            Pa = min(Vf)/tt
            Pp, a = solve(QL =QL, e = q.eta, K = K, Cq = Cq, Cp = Cp, o = q.optmstc, l = Pa )
            print("Round:",tt)
            print( "Policy Regret a*:", mylog( Vavg - Pa ) )
            print( "Policy Regret p*:", mylog( Vavg - Pp ) )
            print( " External Regret:", mylog( Vavg - min(th.mm(Cp, qtavg))) )
            print( "  p*:", a.t() )
            print( "pavg:", ptavg.t() )
            print( "  pt:", pt.t() )
            print( "  qt:", qt.t() )
            print( "d(pt,pavg):", kld( pt, ptavg ) )
            print( "d(p*,pavg):", kld( a, ptavg ) )
            print( "QlossAvg:", th.mm( Cq, ptavg).t() )
            print( "LlossAvg:", th.mm( Cp, qtavg).t() )
            print( "="*50 ) 
        QL = th.cat( ( QL, q.L.clone() ), dim = 1 )
    return True

def LRtuning(args):
    G = th.rand( args.nrand, args.nrand )
    print(G)
    for i in range(1,-9,-1):
        args.lr = 2.0**i
        print(run(args, G=G), 2.0**i)

def MixedFinding(args):
    while not run(args, mix = True):
        continue

parser = argparse.ArgumentParser(description='Set variant algos')
parser.add_argument('--opt', type=int, default=1, dest = 'optimistic')
parser.add_argument('--n', type=int, default=0, dest='nrand')
parser.add_argument('--lr', type=float, default=0.5, dest='lr')
parser.add_argument('--t', type=int, default=100000, dest='T')
parser.add_argument('--z', type=int, default=1, dest='ZeroSum')
args = parser.parse_args()

MixedFinding(args)
