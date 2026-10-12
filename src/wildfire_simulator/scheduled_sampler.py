import math

class ScheduledSampler:
    def __init__(self, k, t0):
        self.k = k
        self.t0 = t0

    def get_prob(self, epoch):
        return 1 / (1 + math.exp(-self.k * (epoch - self.t0)))
