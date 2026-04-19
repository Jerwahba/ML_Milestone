import numpy as np
from src.utils import normalize_fn

class KNN(object):
    """
    kNN classifier object.
    """

    def __init__(self, k=1, task_kind="classification"):
        """
        Call set_arguments function of this class.
        """
        self.k = k
        self.task_kind = task_kind

    def fit(self, training_data, training_labels):
        """
        Trains the model, returns predicted labels for training data.

        Hint: Since KNN does not really have parameters to train, you can try saving
        the training_data and training_labels as part of the class. This way, when you
        call the "predict" function with the test_data, you will have already stored
        the training_data and training_labels in the object.

        Arguments:
            training_data (np.array): training data of shape (N,D)
            training_labels (np.array): labels of shape (N,)
        Returns:
            pred_labels (np.array): labels of shape (N,)
        """

        pred_labels = []

        self.training_labels
        self.means = training_data.mean(0,keepdims=True)
        self.stds  = training_data.std(0,keepdims=True)

        self.normalized_space = normalize_fn(training_data, self.means, self.stds)

        for i in range(self.normalized_space.shape[0]):
            point = self.normalized_space[i]
            diff = self.normalized_space - point
            squared = diff ** 2
            distances = np.sqrt(np.sum(squared, axis=1))
            distances[i] = np.inf
            indices = np.argsort(distances)[:self.k]
            k_nearest_neighbour = training_labels[indices]
            most_common_label = np.argmax(np.bincount(k_nearest_neighbour))
            pred_labels.append(most_common_label)
            # add weights possibly

        pred_labels = np.array(pred_labels)
        return pred_labels

    def predict(self, test_data):
        """
        Runs prediction on the test data.

        Arguments:
            test_data (np.array): test data of shape (N,D)
        Returns:
            test_labels (np.array): labels of shape (N,)
        """
        test_labels = []

        test_data = normalize_fn(test_data, self.means, self.stds)

        for i in range(test_data.shape[0]):
            point = test_data[i]
            diff = self.normalized_space - point
            squared = diff**2
            distances = np.sqrt(np.sum(squared, axis=1))
            indices = np.argsort(distances)[:self.k]

            
            k_nearest_neighbour = self.training_labels[indices]
            most_common_label = np.argmax(np.bincount(k_nearest_neighbour))
            test_labels.append(most_common_label)

            ## question here for tmrw
            ## it is asked in pdf that knn should support regression and classification
            ## here i've implemented classficiation
            ## for regression we look at k nearest neighbours then take their mean
            ## but this is taken from train_labels_reg which is not provided here 
            ## depends on task_kind, should i still implement it, or it's hard coded ?
        test_labels = np.array(test_labels)
        return test_labels