import numpy as np


class Stat:
    def __init__(self):
        self.history = []

    def update(self, value):
        self.history.append(value)

    def avg(self):
        return np.mean(self.history)

    def std(self):
        return np.std(self.history)

    def min(self):
        return np.min(self.history)

    def max(self):
        return np.max(self.history)
