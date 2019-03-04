from math import log

def mylog(x):
    if x < 0:
        return "neg "+str(log(-x))[:5]
    elif x > 0:
        return str(log(x))[:9]
    else:
        return "zero"
