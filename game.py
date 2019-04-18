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
                th.set_printoptions(precision=8, threshold=200, edgeitems=2, linewidth=180, profile=None)
            self.Cp = th.rand( nrand, nrand )
            self.Cq = th.rand( nrand, nrand )

        if zerosum:
            self.Cq = 1 - self.Cp.t() if onesum else -self.Cp.t()
        self.zerosum = zerosum
        self.K = self.Cp.size(0)
        self.avg = {}
        self.sum = {}
        for i in [ 'V', 'pt', 'qt', 'CCE', 'LF', 'BR' ]:
            self.avg[ i ] = 0
        for i in[ 'VL', 'ES' ]:
            self.sum[ i] = 0

        if (self.K, seed) in nashdic:
            self.nash = nashdic[ (self.K, seed) ]
        else:
            self.nash = None
        self.f = open("reglog.txt","w+") if logf else False

    def avgupdate(self, t, **kwargs):
        for i in kwargs:
            self.avg[ i ] = self.avg[ i ] / t * (t-1) + kwargs[ i ] / t
        self.now = kwargs 
    def sumupdate(self, **kwargs):
        for i in kwargs:
            self.sum[ i ] +=  kwargs[ i ]
    
    def log(self, tt):
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
            d = kld(self.nash[0], self.now['pt'] ) + kld(self.nash[1], self.now['qt'] )
            print( "  d(pt, ptavg)", mylog( kld( self.now['pt'], self.avg['pt']) ) )
            print( "d(nash, pt qt)", mylog( d ) )
            if self.f:
                self.f.write( str(tt) + " " + str(float( d )) + "\n")
        print( "="*50 ) 
    
    def  __repr__( self ):
        s = "="*50 + '\n' + "p Loss:\n" + str(self.Cp) 
        return s + "\n q Loss:\n" + str(self.Cq) + "\n" + "="*50 
