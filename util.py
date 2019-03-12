from math import log
import torch as th

def kld(p, q):
    kl = 0.0
    for pi,qi in zip(p,q):
        if pi == 0:
            continue
        kl += pi * log( pi/qi )
    return float(kl)

def l2d(p, q):
    return float(th.norm(p-q))

def mylog(x):
    if x < 0:
        return "neg "+str(log(-x))[:5]
    elif x > 0:
        return str(log(x))[:9]
    else:
        return "zero"
