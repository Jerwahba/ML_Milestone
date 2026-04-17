import numpy as np

from ..utils import get_n_classes, label_to_onehot, onehot_to_label, append_bias_term


class LogisticRegression(object):
    """
    Logistic regression classifier.
    """

    def __init__(self, lr, max_iters=500):
        """
        Initialize the new object (see dummy_methods.py)
        and set its arguments.

        Arguments:
            lr (float): learning rate of the gradient descent
            max_iters (int): maximum number of iterations
        """
        self.lr = lr
        self.max_iters = max_iters

    def softmax(self, z):
        """Calcule les probabilités pour chaque classe de manière stable."""
        # On soustrait le max pour éviter que np.exp(z) ne devienne trop grand (NaN) 
        z_stable = z - np.max(z, axis=1, keepdims=True)
        exp_z = np.exp(z_stable)
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def fit(self, training_data, training_labels):
        """
        Trains the model, returns predicted labels for training data.

        Arguments:
            training_data (np.array): training data of shape (N,D)
            training_labels (np.array): regression target of shape (N,)
        Returns:
            pred_labels (np.array): target of shape (N,)
        """
        ##
        ###
        #### WRITE YOUR CODE HERE!
        ###
        ##

        # J'ajoute le biais et je met les bonnes dimensions aux labels
        X = append_bias_term(training_data)
        y_onehot = label_to_onehot(training_labels)

        N = X.shape[0]
        D_plus_1 = X.shape[1]
        K = y_onehot.shape[1]

        # J'initialise les W a 0
        self.weights = np.zeros((D_plus_1, K))


        for i in range(self.max_iters):
            # calcul score
            logits = np.dot(X, self.weights)

            probs = self.softmax(logits)

            gradient = np.dot(X.T, (probs - y_onehot)) / N

            self.weights -= self.lr * gradient

        return self.predict(training_data)

    def predict(self, test_data):
        """
        Runs prediction on the test data.

        Arguments:
            test_data (np.array): test data of shape (N,D)
        Returns:
            pred_labels (np.array): labels of shape (N,)
        """
        ##
        ###
        #### WRITE YOUR CODE HERE!
        ###
        ##

        # j'ajoute la colonne de 1 (biais)
        X_test = append_bias_term(test_data)

        # je calcul les scores
        logits = np.dot(X_test, self.weights)

        # je transforme en proba avec softmax    
        probs = self.softmax(logits)

        # je prends le plus grand
        pred_labels = np.argmax(probs, axis=1)

        return pred_labels