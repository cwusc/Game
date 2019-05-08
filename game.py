from util import *
from nash import *

class game:
    def __init__(self, nrand = -1, G = None, Gp = None, Gq = None, seed = -1, 
                onesum = False, zerosum = True, logf = True):
        if nrand <= 0:
            self.Cp = th. tensor([
            [0.7745, 0.1880, 0.8293],
            [0.4754, 0.9853, 0.1900],
            [0.4255, 0.5795, 0.9747]])
        elif G is not None:
            self.Cp = G
        elif Gp is not None:
            self.Cp,self.Cq = Gp,Gq
        else:
            if seed > 0:
                th.manual_seed( seed )
                th.set_printoptions(precision=6, threshold=200, edgeitems=2, linewidth=180, profile=None)
            self.Cp = th.rand( nrand, nrand )
            self.Cq = th.rand( nrand, nrand )
        if zerosum:
            self.Cq = 1 - self.Cp.t() if onesum else -self.Cp.t()
        self.zerosum = zerosum
        self.K = self.Cp.size(0)
        self.avg = {}
        self.sum = {}
        self.now = {}
        for i in [ 'V', 'pt', 'qt', 'CCE', 'LF', 'BR', 'ptp', 'qtp' ]:
            self.avg[ i ] = 0
        for i in[ 'VL', 'ES' ]:
            self.sum[ i ] = 0

        if (self.K, seed) in nashdic:
            self.nash = nashdic[ (self.K, seed) ]
            self.value = self.nash[0].t().mm(self.Cp).mm(self.nash[1])
        else:
            self.nash = None
        self.f = open("reglog.txt","w+") if logf else False
        self.klsum = 0
        self.inqsum = 0
        self.eig = float( th.min(th.eig(self.Cp.mm(self.Cp.t()))[0][:,0]) )
    def avgupdate(self, t, **kwargs):
        self.avl = self.avg
        for i in kwargs:
            self.avg[ i ] = self.avg[ i ] / t * (t-1) + kwargs[ i ] / t
        self.las = self.now
        self.now = kwargs 
        if t > 1:
            self.klsum += kld( self.now['ptp'], self.las['ptp'] ) + \
                     kld( self.now['qtp'], self.las['qtp'] ) 
            if self.nash:
                self.inqsum += (self.now['ptp']-self.nash[0]).t().mm( self.Cp ).mm( self.las['qt'] ) + \
                               (self.now['qtp']-self.nash[1]).t().mm( self.Cq ).mm( self.las['pt'] )
    def sumupdate(self, **kwargs):
        for i in kwargs:
            self.sum[ i ] +=  kwargs[ i ]
    def klo(self, x, y):
        s = "kld(self.nash[0],self.{}['p{}']) + kld(self.nash[1],self.{}['q{}'])"
        return float(eval(s.format(x,y,x,y)))
    def pqvalue(self, l, r, x, y):
        ps = "self.{}['p{}'].t().mm(self.Cp).mm(self.{}['q{}'])"   
        qs = "self.{}['q{}'].t().mm(self.Cq).mm(self.{}['p{}'])"
        return float( eval(ps.format(x,l,y,r)) + eval(qs.format(x,l,y,r)) )
    def log(self, tt, p, q):
        #Don't Sum In this Function!
        assert p.eta == q.eta
        et = p.eta
        Rfix = min( th.mm( self.Cp, self.avg[ 'qt'] ) )
        Rswp = sum( th.min( self.avg[ 'LF' ], dim = 1 )[0] ) 
        print("Round:",tt)
        print( "External Regret:", mylog(self.avg['V'] - Rfix) )
        print( "    Swap Regret:", mylog(self.avg['V'] - Rswp) )
        print( " Dynamic Regret:", mylog(self.avg['V'] - self.avg['BR']) ) 
        print( "pavg:", self.avg['pt'].t() )
        print( "qavg:", self.avg['qt'].t() )
        print( "  pt:", self.now['pt'].t() )
        print( "  qt:", self.now['qt'].t()  )
        print( "LlossAvg:", th.mm( self.Cp, self.avg['qt']).t() )
        print( "QlossAvg:", th.mm( self.Cq, self.avg['pt']).t() )
        if self.nash is not None:
            dot = self.klo('now','t')
            dof = self.klo('now','tp')
            dol = self.klo('las','t')
            rem = et*(self.now['ptp']-self.nash[0]).t().mm( self.Cp ).mm( self.las['qt'] ) + \
                  et*(self.now['qtp']-self.nash[1]).t().mm( self.Cq ).mm( self.las['pt'] )
            ratio = dol / dof
            check = H(self.nash[0])+H(self.nash[1])-H(self.now['ptp'])-H(self.now['qtp'])+\
                    et*tt*-self.pqvalue('tp','t','now','avl')
            print( " d(nash, pt qt)", dot )
            print( " d(nash,pt'qt')", dof )
            print( "          check", float(check) )
            print( "sumd(x't,x't-1)", self.klsum  )
            print( "sum<pt'-p*,t-1>", float(self.inqsum*p.eta) )
            print( "(*,t-1')-(*,t')", float( dol-dof ) )
            print( "          ratio", ratio )
            if self.f:
                self.f.write( "{} {:.8E} {:.8E} {:.8E}\n".format(tt, dol-dof, ratio, dof) )
        print( "="*50 ) 
    def  __repr__( self ):
        s = "="*50 + "\np Loss:\n" + str(self.Cp) + "\nq Loss:\n" + str(self.Cq)
        return s + "\nMin sigma: " + str(self.eig) + "\n" + "="*50 