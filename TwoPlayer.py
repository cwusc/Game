import argparse
from player import *
from model import *

def run( args, G = None, Gp = None, Gq = None, mix = False, nash = None ):
    optimistic = not (args.nopt)
    ZeroSum = not (args.nonz)
    nrand = args.nrand
    stint = args.stint
    step = args.step
    th.set_default_tensor_type(th.DoubleTensor)
    reglog = open("reglog.txt", "w+")
    if nrand <= 0:
        # Paper Toss
        #Cp = th.tensor([[0.0,1,-1],[-1,0,1],[1,-1,0]], requires_grad=False)
        # Stop and Go
        #Cp = th.tensor([[0.5,0.5],[0.51,0]], requires_grad=False) #Stop & Go
        #Cq = th.tensor([[0.5,0.51],[0.5,0]], requires_grad=False) #Stop & Go
        #Spin in last iterate convergence
        Cp = th.tensor([[0.1699, 0.3604, 0.0821, 0.2964],
        [0.3091, 0.3913, 0.0825, 0.4189],
        [0.4109, 0.8477, 0.9726, 0.0640],
        [0.3729, 0.3603, 0.8482, 0.1642]])
    elif G is not None:
        Cp = G
    elif Gp is not None:
        Cp,Cq = Gp,Gq
    else:
        if args.seed > 0:
            th.manual_seed( args.seed )
        Cp = th.rand( nrand, nrand )
        Cq = th.rand( nrand, nrand )

    if ZeroSum == 1:
        Cq = 1-Cp.t() if args.bandits else -Cp.t()

    K = Cp.size(0)
    T = args.T

    if args.lr > 0:
        eta = args.lr
    elif optimistic and not args.bandits:
        eta = (log(K)/T)**0.25
        lrf = lambda t: (log(K)/(t))**(2/3)
    else: 
        eta = (log(K)/T)**0.5
        lrf = lambda t: (log(K)/t)**0.5

    print("="*50)
    print("p Loss:\n", Cp)
    print("q Loss:\n", Cq)
    print("="*50)

    th.set_printoptions(precision=4, threshold=200, edgeitems=2, linewidth=180, profile=None)

    if args.swap:
        p = hohplayer(K = K, eta = eta, optimistic = optimistic, bandits = args.bandits )
        q = hohplayer(K = K, eta = eta, optimistic = optimistic, bandits = args.bandits )
    else:
        p = player(K = K, eta = eta, optimistic = optimistic, bandits = args.bandits )
        q = player(K = K, eta = eta, optimistic = optimistic, bandits = args.bandits )

    Vavg = 0
    ptavg = th.zeros(K,1)
    qtavg = th.zeros(K,1)
    Vf = th.zeros(K,1)
    QL = th.zeros(K,1) 
    CCE = th.zeros(K,K)
    LF = th.zeros(K,K)
    BR = 0
    BA = 0
    WPL = 0
    VL = 0 
    ES = 0

    for tt in range(1,T+1):
        if args.dylr:
            eta = lrf( tt )
        if tt > 1:
            WPL += ( l1d(p.play(eta), pt)  )  * tt
        
        pt = p.play(eta)
        qt = q.play(eta)


        q.loss( th.mm(Cq, pt) )
        p.loss( th.mm(Cp, qt) )

        V = th.mm( pt.t(), th.mm(Cp, qt) )
        Vavg = (Vavg/tt)*(tt-1) + V/tt
        ptavg = (ptavg/tt)*(tt-1) + pt/tt
        qtavg = (qtavg/tt)*(tt-1) + qt/tt
        CCE = (CCE/tt)*(tt-1) + th.mm(pt,qt.t())/tt
        LF = (LF/tt)*(tt-1) + th.mm( pt, th.mm(Cp, qt).t() ) / tt
        BR = (BR/tt)*(tt-1) + th.min( th.mm(Cp, qt) )/tt
        VL += l1d( pt, ptavg )**2 
        ES += eta


        if tt == T:
            break
        if args.policy == 1:
            for k in range(K):
                pk = onehot(k,K)
                Vf[k] += th.mm( Cp, q.policyplay( th.mm(Cq, pk) ) )[k]
        if tt > stint*step : #log(tt) >  stint:
            if mix and min(ptavg) < 0.01:
                return False
            stint += 1
            print("Round:",tt)
            Pa = min(Vf)/tt
            Rs = min(th.mm(Cp, qtavg))
            if args.policy == 1:
                Pp, a = solve(QL =QL, e = q.eta, K = K, Cq = Cq, Cp = Cp, 
                               o = q.optmstc, l = Pa, itermin = args.itermin )
                print( "Policy Regret a*:", mylog( Vavg - Pa ) )
                print( "Policy Regret p*:", mylog( Vavg - Pp ) )
                print( " External Regret:", mylog( Vavg - Rs ) )
                print( "  p*:", a.t() )
                print( "d(pt,pavg):", kld( ptavg, pt ) )
                print( "d(p*,pavg):", kld( a, ptavg ) )
            else:
                print( "External Regret:", mylog( Vavg - Rs) )
                print( "    Swap Regret:", mylog( Vavg - sum(th.min( LF, dim = 1 )[0]) ) )
                print( " Dynamic Regret:", mylog( Vavg - BR ) ) 
            print( "pavg:", ptavg.t() )
            print( "qavg:", qtavg.t() )
            print( "  pt:", p.pt.t() )
            print( "  qt:", q.pt.t() )
            print( "LlossAvg:", th.mm( Cp, qtavg).t() )
            print( "QlossAvg:", th.mm( Cq, ptavg).t() )
            if nash is not None:
                #print( "   d(pt,nash):", mylog( l1d(pt,nash)) )
                #print( "   d(nash,pt):", mylog( kld( nash, pt ) ) )
                print( "  d(ptavg,pt):", mylog( kld( nash[0], pt )+kld( nash[1], qt )) )
                #print( "  d(ptavg,pt):", mylog( kld( pt, ptavg )+kld( qt, ptavg )) )
                print( "  Path Length:", mylog( WPL/tt ) )
                print( "      Eta Sum:", mylog( ES ) )
                print( "   R_T + R'_T:", mylog( max(-th.mm(Cq, ptavg))-min(th.mm(Cp, qtavg))) )
                print( "   Var length:", mylog( VL/tt ) )
                print( "lr:", eta )
            if not args.swap:
                print( "p.L:", p.L.t() )
                print( "q.L:", q.L.t() )
            if ZeroSum != 1:
                print( "CCE:\n", CCE ) 
            print( "="*50 ) 
            reglog.write( str(tt)+" "+str( float( kld( nash[0], pt )+kld( nash[1], qt ))) + "\n")

        if args.policy:
            QL = th.cat( ( QL, q.L.clone() ), dim = 1 )
    print("="*50)
    print("p Loss:\n", Cp)
    print("q Loss:\n", Cq)
    print("="*50)
    if log(tt) <  stint:
        return ptavg, Cp
    else:
        return True

def LRtuning(args):
    G = th.rand( args.nrand, args.nrand )
    print(G)
    for i in range(1,-9,-1):
        args.lr = 2.0**i
        print(run(args, G=G), 2.0**i)

def MixedFinding(args):
    while True:
        i = randint(1,1000000)
        args.seed = i
        if run(args, mix = True):
            break
    print("seed:",i)
def LastIterConv(args):
    for i in range(10):
        args.seed = randint(1,10000)
        T = args.T
        args.T = 1000000
        nash, G = run(args, stint = 99999999)
        print("Nash:", nash.t())
        args.T = T
        run(args, nash = nash, G = G)


parser = argparse.ArgumentParser(description='Set variant algos')
parser.add_argument('--n', type=int, default=0, dest='nrand')
parser.add_argument('--lr', type=float, default=-1, dest='lr')
parser.add_argument('--t', type=int, default=100000, dest='T')
parser.add_argument('--iter', type=int, default=1000, dest='itermin')
parser.add_argument('--p', action='store_true', default=False, dest='policy')
parser.add_argument('--b', action='store_true', default=False, dest='bandits')
parser.add_argument('--nopt', action='store_true', default=False, dest='nopt')
parser.add_argument('--nonz', action='store_true', default=False, dest='nonz')
parser.add_argument('--swap', action='store_true', default=False, dest='swap')
parser.add_argument('--dylr', action='store_true', default=False, dest='dylr')
parser.add_argument('--seed', type=int, default=-1, dest='seed')
parser.add_argument('--stint', type=int, default=5, dest='stint')
parser.add_argument('--step', type=int, default=1000, dest='step')


args = parser.parse_args()

run(args, nash = (th.tensor([[0.2627, 0.3057, 0.0549, 0.3767]]).t(),
    th.tensor([[0.1628, 0.4482, 0.1682, 0.2207]]).t() ))
#LastIterConv(args)
#run( args )
#MixedFinding(args)
