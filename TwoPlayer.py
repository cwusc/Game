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
        #Cp = th.tensor([[0.5,0.5],[0.51,0]], requires_grad=False) #Stop & Go
        #Cq = th.tensor([[0.5,0.51],[0.5,0]], requires_grad=False) #Stop & Go
        Cp = th.tensor([[0.2847, 0.9854],[0.5422, 0.1016]])
        Cq = th.tensor([[0.9630, 0.4686],[0.0727, 0.6279]])
        #Cp = th.tensor([[0.8863, 0.1772, 0.9386],
        #[0.2670, 0.9956, 0.9544],
        #[0.6265, 0.5980, 0.1228]])
        #Cq = th.tensor([[0.4271, 0.8174, 0.5112],
        #[0.7327, 0.7630, 0.4404],
        #[0.4437, 0.3071, 0.5904]])
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

    torch.set_printoptions(precision=4, threshold=200, edgeitems=2, linewidth=180, profile=None)

    p = player(K = K, eta = eta, optimistic = optimistic)
    q = player(K = K, eta = eta, optimistic = optimistic)

    Vavg = 0
    ptavg = th.zeros(K,1)
    qtavg = th.zeros(K,1)
    Vf = th.zeros(K,1)
    QL = q.L.clone()
    CCE = th.zeros(K,K)
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
        CCE = (CCE/tt)*(tt-1) + th.mm(pt,qt.t())/tt

        if tt == T:
            break
        if args.policy == 1:
            for k in range(K):
                pk = th.tensor( [[ float(j==k) for j in range(K)]] ).t()
                Vf[k] += th.mm( Cp, q.policyplay( th.mm(Cq, pk) ) )[k]
        if tt > 1000 * stint:
            if mix and min(ptavg) < 0.01:
                return False
            stint += 1
            print("Round:",tt)
            Pa = min(Vf)/tt
            if args.policy == 1:
                Pp, a = solve(QL =QL, e = q.eta, K = K, Cq = Cq, Cp = Cp, 
                               o = q.optmstc, l = Pa, itermin = args.itermin )
                print( "Policy Regret a*:", mylog( Vavg - Pa ) )
                print( "Policy Regret p*:", mylog( Vavg - Pp ) )
                print( " External Regret:", mylog( Vavg - min(th.mm(Cp, qtavg))) )
                print( "  p*:", a.t() )
                print( "d(pt,pavg):", kld( pt, ptavg ) )
                print( "d(p*,pavg):", kld( a, ptavg ) )
            print( "pavg:", ptavg.t() )
            print( "qavg:", qtavg.t() )
            print( "  pt:", pt.t() )
            print( "  qt:", qt.t() )
            print( "LlossAvg:", th.mm( Cp, qtavg).t() )
            print( "QlossAvg:", th.mm( Cq, ptavg).t() )
            print( "CCE:\n", CCE ) 
            print( "="*50 ) 

        if args.policy == 1:
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
parser.add_argument('--iter', type=int, default=1000, dest='itermin')
parser.add_argument('--p', type=int, default=0, dest='policy')
args = parser.parse_args()

run(args)
