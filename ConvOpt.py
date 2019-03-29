import argparse
from player import *
from adversary import *

def run( args, G = None, Gp = None, Gq = None, mix = False):
    optimistic = not (args.nopt)
    K = args.nrand
    T = args.T
    th.set_default_tensor_type(th.DoubleTensor)
    reglog = open("reglog.txt", "w+")

    if args.lr > 0:
        eta = args.lr

    th.set_printoptions(precision=4, threshold=200, edgeitems=2, linewidth=180, profile=None)

    if args.swap:
        p = metaplayer(K = K, eta = eta, optimistic = optimistic, bandits = args.bandits )
    else:
        p = player( K = K, eta = eta, optimistic = optimistic, bandits = args.bandits )
    q = adversary( K = K )
    
    L = 0
    LF = th.zeros(K,K)
    ptavg = th.zeros(K,1)
    ltavg = th.zeros(K,1)
    stint = 5

    for tt in range(1,T+1):
        if args.bandits:
            eta = (log(K)/(K*tt))**0.5
        pt = p.play(eta)
        lt = q.play(pt)

        p.loss( lt )

        
        ptavg = (ptavg/tt)*(tt-1) + pt / tt
        ltavg = (ltavg/tt)*(tt-1) + lt / tt
        L = (L/tt)*(tt-1) + th.mm( lt.t(), pt ) / tt 
        LF = (LF/tt)*(tt-1) + th.mm( pt, lt.t() ) / tt

        if log(tt) >  stint:
            if mix and min(ptavg) < 0.002:
                return False
            stint += 0.1
            print("Round:",tt)
            print( "External Regret:", mylog( L - min(ltavg) ) )
            print( "    Swap Regret:", mylog( L - sum(th.min( LF, dim = 1 )[0]) ) )
            print( "="*50 ) 
            reglog.write( str(tt) + " " + str(float(L - min(ltavg))) + "\n")

    return True


parser = argparse.ArgumentParser(description='Set variant algos')
parser.add_argument('--n', type=int, default=0, dest='nrand')
parser.add_argument('--lr', type=float, default=0.5, dest='lr')
parser.add_argument('--t', type=int, default=100000, dest='T')
parser.add_argument('--b', action='store_true', default=False, dest='bandits')
parser.add_argument('--nopt', action='store_true', default=False, dest='nopt')
parser.add_argument('--swap', action='store_true', default=False, dest='swap')
parser.add_argument('--seed', type=int, default=-1, dest='seed')

args = parser.parse_args()

run(args)
