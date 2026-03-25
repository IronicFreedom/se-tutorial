import numpy as np
import matplotlib.pyplot as plt

class VicsekModel:
    def __init__(self, n):
        self.n = n
        self.r = np.random.random((self.n, 2))
        self.theta = np.random.random(self.n)

        self.d = 0.01
        self.v = 0.01
        self.dt = 1
        self.eta = 0.1

    @staticmethod
    def distance(p1, p2):
        return np.sqrt(((p1 - p2) ** 2).sum())
    
    def update_model(self):
        for i in range(self.n):
            sum_sin = 0
            sum_cos = 0
            neighbours = 0

            for j in range(self.n):
                if i != j:
                    if distance(r[i], r[j]) < self.d:
                        theta_j = 2 * np.pi * self.theta[j]
                        sum_sin = sum_sin + np.sin(theta_j)
                        sum_cos = sum_cos + np.cos(theta_j)
                        neighbours = neighbours + 1

            if neighbours > 0:
                avg_theta = np.arctan2(sum_sin / neighbours, sum_cos / neighbours)
                self.theta[i] = (avg_theta / (2 * np.pi)) + self.eta * (np.random.rand() - 0.5)

            dx = self.v * self.dt * np.cos(2 * np.pi * self.theta[i])
            dy = self.v * self.dt * np.sin(2 * np.pi * self.theta[i])

            self.r[i, 0] = self.r[i, 0] + dx
            self.r[i, 1] = self.r[i, 1] + dy

            r %= 1