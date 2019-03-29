from random import random
import torch as th
class adversary:
    def __init__ (self, K = 2, oblivious = True):
        self.K = K
        self.obl = oblivious
    def play(self, pt = None ):
        if self.obl:
            return th.tensor([[ random() for i in range(self.K)]]).t()
