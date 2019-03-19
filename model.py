import torch as th
from torch import nn
import torch.optim

th.set_default_tensor_type(th.DoubleTensor)

class Model(nn.Module):
    def __init__( self, eta, K, Cp, Cq, optimistic = False, det = False):
        super(Model, self).__init__()
        self.eta = eta
        self.sigma = nn.Softmax(dim=0)
        if det:
            self.ar = th.ones(K, 1, requires_grad=True)
        else:
            self.ar = th.randn(K, 1, requires_grad=True)
        self.optmstc = optimistic
        self.Cp = Cp
        self.Cq = Cq

    def forward(self, x, aa = None):
        a = self.sigma(self.ar) if aa is None else aa
        La = th.mm( self.Cq, a)
        if self.optmstc:
            La = La * 2
        Qta = self.sigma( -self.eta * ( x + La ) ) 
        DeviateLossVec = th.mm( self.Cp, Qta )
        return th.mean( th.mm( a.t(), DeviateLossVec ) )


def solve(QL, e, K, Cq, Cp, o, l, itermin=1000):
    md = Model(eta = e, K = K, Cq = Cq, Cp = Cp, optimistic = o)
    optimizer = torch.optim.RMSprop( [md.ar] )
    QL.requires_grad = False
    eps = 1e-6
    it = 1
    while it < itermin or P > l + eps:
        optimizer.zero_grad()
        P = md( QL )
        P.backward()
        optimizer.step()
        it += 1
        if it > 10*itermin:
            break
    print("Solver iterations:",it)
    Pd = P.detach()
    od = md.sigma( md.ar ).detach()
    return Pd, od
