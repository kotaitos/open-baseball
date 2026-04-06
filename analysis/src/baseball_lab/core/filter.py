import math

class OneEuroFilter:
    """高速な動きへの追従と低速時のノイズ除去を両立するフィルタ"""
    def __init__(self, freq, mincutoff=1.0, beta=0.007, dcutoff=1.0):
        self.freq = freq
        self.mincutoff = mincutoff
        self.beta = beta
        self.dcutoff = dcutoff
        self.x_prev = None
        self.dx_prev = 0

    def _alpha(self, cutoff):
        tau = 1.0 / (2 * math.pi * cutoff)
        te = 1.0 / self.freq
        return 1.0 / (1.0 + tau / te)

    def filter(self, x):
        if self.x_prev is None:
            self.x_prev = x
            return x
        
        dx = (x - self.x_prev) * self.freq
        edx = self.dx_prev + (self._alpha(self.dcutoff) * (dx - self.dx_prev))
        cutoff = self.mincutoff + self.beta * abs(edx)
        
        result = self.x_prev + (self._alpha(cutoff) * (x - self.x_prev))
        self.x_prev = result
        self.dx_prev = edx
        return result
