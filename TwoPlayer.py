from argpar import *
from player import *
from model import *
from game import *

def run( args, G = None, Gp = None, Gq = None, mix = False, nash = None ):
    optimistic = not (args.nopt)
    g = game( args.nrand, G, Gp, Gq, args.seed, bool(args.bandits), not(args.nonz), not(mix))
    K = g.Cp.size(0)
    T = args.T
    if not mix:
        print(g)

    if optimistic and not args.bandits:
        lrf = lambda t: (log(K)/(t))**(1/4)
    elif args.swap:
        lrf = lambda t: (K*log(K)/t)**0.5
    else: 
        lrf = lambda t: (log(K)/t)**0.5
    eta = args.lr if args.lr > 0 else lrf(T)

    th.set_printoptions(precision=4, threshold=200, edgeitems=2, linewidth=180, profile=None)

    if args.swap:
        p = metaplayer(K = K, eta = eta, optimistic = optimistic, bandits = args.bandits )
        q = metaplayer(K = K, eta = eta, optimistic = optimistic, bandits = args.bandits )
    else:
        p = player(K = K, eta = eta, optimistic = optimistic, bandits = args.bandits )
        q = player(K = K, eta = eta, optimistic = optimistic, bandits = args.bandits )
    if args.policy:
        Vf = th.zeros(K,1)
        QL = th.zeros(K,1) 

    for tt in range(1,T+1):
        if args.dylr:
            eta = lrf( tt )
        pt = p.play(eta)
        qt = q.play(eta)
        q.loss( th.mm( g.Cq, pt) )
        p.loss( th.mm(g.Cp, qt) )

        if not mix:
            g.avgupdate( tt, V = th.mm( pt.t(), th.mm(g.Cp, qt) ) , 
                    pt = pt, qt = qt, CCE = th.mm(pt,qt.t()), 
                    LF = th.mm( pt, th.mm(g.Cp, qt).t() ), 
                    BR = th.min( th.mm(g.Cp, qt) ) )
            g.sumupdate( VL = l1d( pt, g.avg['pt'] )**2, ES = eta )

        if tt == T:
            break
        if args.policy == 1:
            for k in range(K):
                pk = onehot(k,K)
                Vf[k] += th.mm( g.Cp, q.policyplay( th.mm( g.Cq, pk) ) )[k]
        if tt > args.stint * args.step : #log(tt) >  stint:
            args.stint += 1
            if mix and ( min( pt ) < .01 or min( qt ) < .01 ):
                return False
            if not mix:
                g.log(tt)

        if args.policy:
            QL = th.cat( ( QL, q.L.clone() ), dim = 1 )
    print(g)
    return True
def MixedFinding(args):
    c = 0
    while True:
        i = randint(1,99999999)
        args.seed = i
        if run(args, mix = True):
            break
        c += 1
        if c % 100 == 0:
            print(c,"games tried")
    print("seed:",i)
args =  getargs()
if args.mixf:
    MixedFinding(args)
else:
    run( args )
