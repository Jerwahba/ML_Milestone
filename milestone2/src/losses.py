import numpy as np

class MSE:
    @staticmethod
    def loss(y_true, y_pred):
        """
        :param y_true: (array) One hot encoded truth vector.
        :param y_pred: (array) Prediction vector
        :return: (flt)
        """
        return np.mean((y_true - y_pred) ** 2)

    @staticmethod
    def gradient(y_true, y_pred):
        return 2.0 * (y_pred - y_true) / y_true.size

class CrossEntropy:
    @staticmethod
    def loss(y_true, y_pred):
        """
        :param y_true: (array) One hot encoded truth vector.
        :param y_pred: (array) Prediction vector
        :return: (flt)
        """
        eps = 1e-15
        y_pred_clipped = np.clip(y_pred, eps, 1.0 - eps)
        return -np.mean(np.sum(y_true * np.log(y_pred_clipped), axis=1))

    @staticmethod
    def gradient(y_true, y_pred):
        eps = 1e-15
        y_pred_clipped = np.clip(y_pred, eps, 1.0 - eps)
        return -(y_true / y_pred_clipped) / y_true.shape[0]
