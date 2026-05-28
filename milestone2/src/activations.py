import numpy as np

class Sigmoid:
    @staticmethod
    def forward(z):
        # Clip z to prevent overflow in exp
        clipped_z = np.clip(z, -500, 500)
        return 1.0 / (1.0 + np.exp(-clipped_z))

    @staticmethod
    def gradient(z):
        s = Sigmoid.forward(z)
        return s * (1.0 - s)

class ReLU:
    @staticmethod
    def forward(z):
        return np.maximum(0.0, z)

    @staticmethod
    def gradient(z):
        return (z > 0.0).astype(float)

class Identity:
    @staticmethod
    def forward(z):
        return z

    @staticmethod
    def gradient(z):
        return np.ones_like(z)