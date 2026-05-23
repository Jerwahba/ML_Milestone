import numpy as np

class MLP:
    def __init__(self, dimensions, activations):
        """
        :param dimensions: list of dimensions of the neural net. (input, hidden layer, ... ,hidden layer, output)
        :param activations: list of activation functions. Must contain N-1 activation function, where N = len(dimensions).

        Example of one hidden layer with
        - 2 inputs
        - 10 hidden nodes
        - 5 outputs
        layers -->    [0,        1,          2]
        ----------------------------------------
        dimensions =  (2,     10,          5)
        activations = (      Sigmoid,      Sigmoid)
        """
        self.dimensions = dimensions
        self.activations = activations
        self.num_layers = len(dimensions) - 1
        
        self.weights = {}
        self.biases = {}
        
        # Xavier / Glorot Initialization
        for i in range(self.num_layers):
            dim_in = dimensions[i]
            dim_out = dimensions[i+1]
            limit = np.sqrt(6.0 / (dim_in + dim_out))
            self.weights[i+1] = np.random.uniform(-limit, limit, (dim_in, dim_out))
            self.biases[i+1] = np.zeros((1, dim_out))

    def feed_forward(self, x):
        """
        Execute a forward feed through the network.
        :param x: (array) Batch of input data vectors.
        :return: (tpl) Node outputs and activations per layer. The numbering of the output is equivalent to the layer numbers.
        """
        a = {}
        z = {0: x}
        
        for i in range(1, self.num_layers + 1):
            a[i] = z[i-1] @ self.weights[i] + self.biases[i]
            activation_func = self.activations[i-1]
            z[i] = activation_func.forward(a[i])
            
        return z, a

    def predict(self, x):
        """
        :param x: (array) Containing parameters
        :return: (array) A 2D array of shape (n_cases, n_classes).
        """
        z, a = self.feed_forward(x)
        return z[self.num_layers]

    def back_prop(self, z, a, y_true, loss):
        """
        The input dicts keys represent the layers of the net.
        a = { 0: x,
              1: f(w1(x) + b1)
              2: f(w2(a2) + b2)
              }
        :param a: (dict) w^T@x + b
        :param z: (dict) f(a)
        :param y_true: (array) One hot encoded truth vector.
        :param loss: Loss class with a static .gradient(y_true, y_pred) method.
        :return:
        """
        L = self.num_layers
        y_pred = z[L]
        
        # Loss gradient w.r.t prediction
        loss_grad = loss.gradient(y_true, y_pred)
        
        # Output delta: delta^(L) = loss_grad * f'_L(a^(L))
        activation_func = self.activations[L-1]
        delta = loss_grad * activation_func.gradient(a[L])
        
        for i in range(L, 0, -1):
            # Compute gradient for weights of layer i
            dw = z[i-1].T @ delta
            
            # Save delta for update_w_b
            current_delta = delta
            
            if i > 1:
                # Compute delta for layer i-1
                next_activation_func = self.activations[i-2]
                delta = (current_delta @ self.weights[i].T) * next_activation_func.gradient(a[i-1])
                
            # Update weights and biases for layer i
            self.update_w_b(i, dw, current_delta)

    def update_w_b(self, index, dw, delta):
        """
        Update weights and biases.
        :param index: (int) Number of the layer
        :param dw: (array) Partial derivatives
        :param delta: (array) Delta error.
        """
        self.weights[index] -= self.lr * dw
        self.biases[index] -= self.lr * np.sum(delta, axis=0, keepdims=True)

    def fit(self, x, y_true, loss, epochs, batch_size, learning_rate=1e-3):
        """
        :param x: (array) Containing parameters
        :param y_true: (array) Containing one hot encoded labels.
        :param loss: Loss class (MSE, CrossEntropy etc.)
        :param epochs: (int) Number of epochs.
        :param batch_size: (int)
        :param learning_rate: (flt)
        """
        self.lr = learning_rate
        n_samples = x.shape[0]
        
        for epoch in range(epochs):
            # Shuffle data at the start of each epoch
            indices = np.arange(n_samples)
            np.random.shuffle(indices)
            x_shuffled = x[indices]
            y_shuffled = y_true[indices]
            
            for start_idx in range(0, n_samples, batch_size):
                end_idx = min(start_idx + batch_size, n_samples)
                x_batch = x_shuffled[start_idx:end_idx]
                y_batch = y_shuffled[start_idx:end_idx]
                
                z, a = self.feed_forward(x_batch)
                self.back_prop(z, a, y_batch, loss)

