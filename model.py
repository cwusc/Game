import torch as th
from torch import nn
import torch.optim

th.set_default_tensor_type(th.DoubleTensor)

class Model(nn.Module):
    def __init__( self, eta, K, U, optimistic = False, det = True):
        super(Model, self).__init__()
        self.eta = eta
        self.sigma = nn.Softmax(dim=0)
        if det:
            self.ar = th.zeros(K, 1, requires_grad=True)
        else:
            self.ar = th.randn(K, 1, requires_grad=True)
        self.optmstc = optimistic
        self.U = U

    def forward(self, x):
        if self.optmstc:
            a = self.sigma(self.ar) * 2 
        else:
            a = self.sigma(self.ar) * 1
        La = -th.mm( a.t(), self.U ).t()
        Pta = self.sigma( -self.eta * ( x + La ) ) 
        Qt = th.mm( self.U, Pta )
        return th.sum( th.mm( a.t(), Qt ) )


def solve(QL, e, K, U, o):
    md = Model(eta = e, K = K, U = U, optimistic = 0)
    optimizer = torch.optim.RMSprop( [md.ar] )
    QL.requires_grad = False

    for qq in range(1000):
        optimizer.zero_grad()
        R = md( QL )
        R.backward()
        optimizer.step()
    Rd = R.detach()
    ad = md.sigma( md.ar ).detach()
    return Rd, ad
