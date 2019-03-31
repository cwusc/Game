from math import log
import torch as th
from random import random
from random import randint
from os import system

def sample( p ):
    x = random()
    c = 0
    for i in range(p.size(0)):
        c += p[i]
        if p[i] > 0 and c > x:
            break
    return min(i, p.size(0)-1)

def onehot(i, K):
    x = th.zeros(K,1)
    x[ i ] = 1.0
    return x 

def kld(p, q):
    kl = 0.0
    for pi,qi in zip(p,q):
        if pi == 0:
            continue
        kl += pi * log( pi/qi )
    return float(kl)

def l2d(p, q):
    return float(th.norm(p-q))
def l1d(p, q):
    return float(th.norm(p-q,p=1))

def mylog(x):
    if x < 0:
        return "neg "+str(log(-x))[:5]
    elif x > 0:
        return str(log(x))[:9]
    else:
        return "zero"
