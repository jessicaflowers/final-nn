# Imports
import numpy as np
from typing import List, Dict, Tuple, Union
from numpy.typing import ArrayLike

class NeuralNetwork:
    """
    This is a class that generates a fully-connected neural network.

    Parameters:
        nn_arch: List[Dict[str, float]]
            A list of dictionaries describing the layers of the neural network.
            e.g. [{'input_dim': 64, 'output_dim': 32, 'activation': 'relu'}, {'input_dim': 32, 'output_dim': 8, 'activation:': 'sigmoid'}]
            will generate a two-layer deep network with an input dimension of 64, a 32 dimension hidden layer, and an 8 dimensional output.
        lr: float
            Learning rate (alpha).
        seed: int
            Random seed to ensure reproducibility.
        batch_size: int
            Size of mini-batches used for training.
        epochs: int
            Max number of epochs for training.
        loss_function: str
            Name of loss function.

    Attributes:
        arch: list of dicts
            (see nn_arch above)
    """

    def __init__(
        self,
        nn_arch: List[Dict[str, Union[int, str]]],
        lr: float,
        seed: int,
        batch_size: int,
        epochs: int,
        loss_function: str
    ):

        # Save architecture
        self.arch = nn_arch

        # Save hyperparameters
        self._lr = lr
        self._seed = seed
        self._epochs = epochs
        self._loss_func = loss_function
        self._batch_size = batch_size

        # Initialize the parameter dictionary for use in training
        self._param_dict = self._init_params()

    def _init_params(self) -> Dict[str, ArrayLike]:
        """
        DO NOT MODIFY THIS METHOD! IT IS ALREADY COMPLETE!

        This method generates the parameter matrices for all layers of
        the neural network. This function returns the param_dict after
        initialization.

        Returns:
            param_dict: Dict[str, ArrayLike]
                Dictionary of parameters in neural network.
        """

        # Seed NumPy
        np.random.seed(self._seed)

        # Define parameter dictionary
        param_dict = {}

        # Initialize each layer's weight matrices (W) and bias matrices (b)
        for idx, layer in enumerate(self.arch):
            layer_idx = idx + 1
            input_dim = layer['input_dim']
            output_dim = layer['output_dim']
            param_dict['W' + str(layer_idx)] = np.random.randn(output_dim, input_dim) * 0.1
            param_dict['b' + str(layer_idx)] = np.random.randn(output_dim, 1) * 0.1

        return param_dict

    def _single_forward(
        self,
        W_curr: ArrayLike,
        b_curr: ArrayLike,
        A_prev: ArrayLike,
        activation: str
    ) -> Tuple[ArrayLike, ArrayLike]:
        """
        This method is used for a single forward pass on a single layer.

        Args:
            W_curr: ArrayLike
                Current layer weight matrix.
            b_curr: ArrayLike
                Current layer bias matrix.
            A_prev: ArrayLike
                Previous layer activation matrix.
            activation: str
                Name of activation function for current layer.

        Returns:
            A_curr: ArrayLike
                Current layer activation matrix.
            Z_curr: ArrayLike
                Current layer linear transformed matrix.
        """
        # Compute the linear transformation for this layer:
        #   Z = W * A_prev + b
        # where W is the weight matrix 
        # A_prev is the activation from the previous layer 
        # and b is the bias vector 
        Z_curr = np.dot(W_curr, A_prev) + b_curr

        # Apply sigmoid/relu to Z to get the activation A:
        #   A = g(Z)
        # g is either sigmoid or relu 
        if activation == 'sigmoid': # 1 / (1 + e^(-z))
            A_curr = self._sigmoid(Z_curr)
        elif activation == 'relu': # max(0, z)
            A_curr = self._relu(Z_curr)
        else:
            raise ValueError(f"Unsupported activation function: {activation}")

        return A_curr, Z_curr

    def forward(self, X: ArrayLike) -> Tuple[ArrayLike, Dict[str, ArrayLike]]:
        """
        This method is responsible for one forward pass of the entire neural network.

        Args:
            X: ArrayLike
                Input matrix with shape [batch_size, features].

        Returns:
            output: ArrayLike
                Output of forward pass.
            cache: Dict[str, ArrayLike]:
                Dictionary storing Z and A matrices from `_single_forward` for use in backprop.
        """
        # stores all intermediate A (activation) and Z (linear transform) matrices
        # these are needed later during backprop to compute gradients
        cache = {}

        # network convention is that each column is one sample, so i need
        # the features along rows and samples along columns
        A_curr = X.T
        cache['A0'] = A_curr  # store the input as "activation 0" for use in backprop

        # forward pass through each layer 
        # for each layer l, compute:
        #   Z_l = W_l * A_{l-1} + b_l     
        #   A_l = g_l(Z_l)                 
        for idx, layer in enumerate(self.arch):
            layer_idx = idx + 1
            A_prev = A_curr

            # get parameters for this layer
            W_curr = self._param_dict['W' + str(layer_idx)]
            b_curr = self._param_dict['b' + str(layer_idx)]
            activation = layer['activation']

            # do the single-layer forward pass
            A_curr, Z_curr = self._single_forward(W_curr, b_curr, A_prev, activation)

            # store this layer's A and Z in the cache
            cache['A' + str(layer_idx)] = A_curr
            cache['Z' + str(layer_idx)] = Z_curr

        # output the final layer's activation (ie network output) and the full cache
        return A_curr, cache

    def _single_backprop(
        self,
        W_curr: ArrayLike,
        b_curr: ArrayLike,
        Z_curr: ArrayLike,
        A_prev: ArrayLike,
        dA_curr: ArrayLike,
        activation_curr: str
    ) -> Tuple[ArrayLike, ArrayLike, ArrayLike]:
        """
        This method is used for a single backprop pass on a single layer.

        Args:
            W_curr: ArrayLike
                Current layer weight matrix.
            b_curr: ArrayLike
                Current layer bias matrix.
            Z_curr: ArrayLike
                Current layer linear transform matrix.
            A_prev: ArrayLike
                Previous layer activation matrix.
            dA_curr: ArrayLike
                Partial derivative of loss function with respect to current layer activation matrix.
            activation_curr: str
                Name of activation function of layer.

        Returns:
            dA_prev: ArrayLike
                Partial derivative of loss function with respect to previous layer activation matrix.
            dW_curr: ArrayLike
                Partial derivative of loss function with respect to current layer weight matrix.
            db_curr: ArrayLike
                Partial derivative of loss function with respect to current layer bias matrix.
        """
        m = A_prev.shape[1] #  number of samples in the mini-batch  

        # calc dZ for this layer  
        # dZ = dA * g'(Z)
        # sigmoid:  g'(Z) = sigmoid(Z) * (1 - sigmoid(Z))
        # ReLU:     g'(Z) = 1 if Z > 0, else 0
        if activation_curr == 'sigmoid':
            dZ_curr = self._sigmoid_backprop(dA_curr, Z_curr)
        elif activation_curr == 'relu':
            dZ_curr = self._relu_backprop(dA_curr, Z_curr)
        else:
            raise ValueError(f"Unsupported activation function: {activation_curr}")

        # calc gradient of the loss wrt weights
        # dW = (1/m) * dZ * A_prev^T
        dW_curr = np.dot(dZ_curr, A_prev.T) / m

        # calc gradient of the loss wrt biases
        # db = (1/m) * sum(dZ, axis=1)
        db_curr = np.sum(dZ_curr, axis=1, keepdims=True) / m

        # calc gradient of the loss wrt previous layer's activation 
        # dA_prev = W^T * dZ
        dA_prev = np.dot(W_curr.T, dZ_curr)

        return dA_prev, dW_curr, db_curr

    def backprop(self, y: ArrayLike, y_hat: ArrayLike, cache: Dict[str, ArrayLike]):
        """
        This method is responsible for the backprop of the whole fully connected neural network.

        Args:
            y (array-like):
                Ground truth labels.
            y_hat: ArrayLike
                Predicted output values.
            cache: Dict[str, ArrayLike]
                Dictionary containing the information about the
                most recent forward pass, specifically A and Z matrices.

        Returns:
            grad_dict: Dict[str, ArrayLike]
                Dictionary containing the gradient information from this pass of backprop.
        """
        grad_dict = {} # initialize dicttionary

        # calc dA for the output layer by taking the derivative of the
        # loss function w.r.t the network's final output (y_hat = A_L).
        # BCE: dA_L = -(y / y_hat) + (1 - y) / (1 - y_hat)
        # MSE: dA_L = 2 * (y_hat - y) / m
        if self._loss_func == 'bce':
            dA_curr = self._binary_cross_entropy_backprop(y, y_hat)
        elif self._loss_func == 'mse':
            dA_curr = self._mean_squared_error_backprop(y, y_hat)
        else:
            raise ValueError(f"Unsupported loss function: {self._loss_func}")

        # step backward through the layers. at each layer l use the cached 
        # Z_l and A_{l-1} from the forward pass to calc
        #   dZ_l    = dA_l * g'(Z_l)           (how loss changes wrt Z)
        #   dW_l    = (1/m) * dZ_l * A_{l-1}^T (gradient of the weights)
        #   db_l    = (1/m) * sum(dZ_l)         (bias gradient)
        #   dA_{l-1}= W_l^T * dZ_l             (propagate error to prev layer)
        for idx, layer in reversed(list(enumerate(self.arch))):
            layer_idx = idx + 1

            # get params and cached values for this layer
            W_curr = self._param_dict['W' + str(layer_idx)]
            b_curr = self._param_dict['b' + str(layer_idx)]
            Z_curr = cache['Z' + str(layer_idx)]
            A_prev = cache['A' + str(idx)]  # A from the layer before this one
            activation_curr = layer['activation']

            # calc gradients 
            dA_prev, dW_curr, db_curr = self._single_backprop(
                W_curr, b_curr, Z_curr, A_prev, dA_curr, activation_curr
            )

            # store the weight and bias gradients 
            grad_dict['dW' + str(layer_idx)] = dW_curr
            grad_dict['db' + str(layer_idx)] = db_curr

            # pass the error backward to next layer
            dA_curr = dA_prev

        return grad_dict

    def _update_params(self, grad_dict: Dict[str, ArrayLike]):
        """
        This function updates the parameters in the neural network after backprop. This function
        only modifies internal attributes and does not return anything

        Args:
            grad_dict: Dict[str, ArrayLike]
                Dictionary containing the gradient information from most recent round of backprop.
        """
        # Gradient descent parameter update
        #   W_l = W_l - lr * dW_l
        #   b_l = b_l - lr * db_l
        # lr is the learning rate. subtract the gradient scaled by lr to 
        # move the parameters in the direction that reduces the loss
        for idx in range(len(self.arch)):
            layer_idx = idx + 1
            self._param_dict['W' + str(layer_idx)] -= self._lr * grad_dict['dW' + str(layer_idx)]
            self._param_dict['b' + str(layer_idx)] -= self._lr * grad_dict['db' + str(layer_idx)]

    def fit(
        self,
        X_train: ArrayLike,
        y_train: ArrayLike,
        X_val: ArrayLike,
        y_val: ArrayLike
    ) -> Tuple[List[float], List[float]]:
        """
        This function trains the neural network by backpropagation for the number of epochs defined at
        the initialization of this class instance.

        Args:
            X_train: ArrayLike
                Input features of training set.
            y_train: ArrayLike
                Labels for training set.
            X_val: ArrayLike
                Input features of validation set.
            y_val: ArrayLike
                Labels for validation set.

        Returns:
            per_epoch_loss_train: List[float]
                List of per epoch loss for training set.
            per_epoch_loss_val: List[float]
                List of per epoch loss for validation set.
        """
        np.random.seed(self._seed)

        # track loss at each epoch 
        per_epoch_loss_train = []
        per_epoch_loss_val = []

        for epoch in range(self._epochs):
            # shuffle training data at the start of each epoch 
            indices = np.arange(X_train.shape[0])
            np.random.shuffle(indices)
            X_shuffled = X_train[indices]
            y_shuffled = y_train[indices]

            # keep track of losses for each batch to average at end of epoch
            epoch_losses = []

            # batch gradient descent 
            for i in range(0, X_shuffled.shape[0], self._batch_size):
                # slice out the current batch
                X_batch = X_shuffled[i:i + self._batch_size]
                y_batch = y_shuffled[i:i + self._batch_size]

                # forward pass
                y_hat, cache = self.forward(X_batch)

                # calc loss
                if self._loss_func == 'bce':
                    loss = self._binary_cross_entropy(y_batch.T, y_hat)
                elif self._loss_func == 'mse':
                    loss = self._mean_squared_error(y_batch.T, y_hat)
                else:
                    raise ValueError(f"Unsupported loss function: {self._loss_func}")
                epoch_losses.append(loss)

                # backward pass
                grad_dict = self.backprop(y_batch.T, y_hat, cache)

                # update all weights and biases 
                self._update_params(grad_dict)

            # average training loss across all batches 
            per_epoch_loss_train.append(np.mean(epoch_losses))

            # validation loss ...
            # run forward pass on validation set -> how well the model generalizes to unseen data
            y_val_hat, _ = self.forward(X_val)
            if self._loss_func == 'bce':
                val_loss = self._binary_cross_entropy(y_val.T, y_val_hat)
            elif self._loss_func == 'mse':
                val_loss = self._mean_squared_error(y_val.T, y_val_hat)
            per_epoch_loss_val.append(val_loss)

        return per_epoch_loss_train, per_epoch_loss_val

    def predict(self, X: ArrayLike) -> ArrayLike:
        """
        This function returns the prediction of the neural network.

        Args:
            X: ArrayLike
                Input data for prediction.

        Returns:
            y_hat: ArrayLike
                Prediction from the model.
        """
        # forward pass through network and return only the output
        y_hat, _ = self.forward(X)
        return y_hat

    def _sigmoid(self, Z: ArrayLike) -> ArrayLike:
        """
        Sigmoid activation function.

        Args:
            Z: ArrayLike
                Output of layer linear transform.

        Returns:
            nl_transform: ArrayLike
                Activation function output.
        """
        # squashe any real-valued input into the range (0, 1)
        # sigma(z) = 1 / (1 + e^(-z))
        return 1 / (1 + np.exp(-Z))

    def _sigmoid_backprop(self, dA: ArrayLike, Z: ArrayLike):
        """
        Sigmoid derivative for backprop.

        Args:
            dA: ArrayLike
                Partial derivative of previous layer activation matrix.
            Z: ArrayLike
                Output of layer linear transform.

        Returns:
            dZ: ArrayLike
                Partial derivative of current layer Z matrix.
        """
        # derivative of sigmoid function
        #   d(sigma)/dZ = sigma(Z) * (1 - sigma(Z))
        
        # dZ = dA * d(sigma)/dZ:
        # dZ = dA * sigma(Z) * (1 - sigma(Z))
        sig = self._sigmoid(Z)
        return dA * sig * (1 - sig)

    def _relu(self, Z: ArrayLike) -> ArrayLike:
        """
        ReLU activation function.

        Args:
            Z: ArrayLike
                Output of layer linear transform.

        Returns:
            nl_transform: ArrayLike
                Activation function output.
        """
        #   ReLU(z) = max(0, z)
        # outputs z directly if z > 0, otherwise outputs 0.
        return np.maximum(0, Z)

    def _relu_backprop(self, dA: ArrayLike, Z: ArrayLike) -> ArrayLike:
        """
        ReLU derivative for backprop.

        Args:
            dA: ArrayLike
                Partial derivative of previous layer activation matrix.
            Z: ArrayLike
                Output of layer linear transform.

        Returns:
            dZ: ArrayLike
                Partial derivative of current layer Z matrix.
        """
        # derivative of ReLU is piecewise function
        #   d(ReLU)/dZ = 1  if Z > 0
        #   d(ReLU)/dZ = 0  if Z <= 0
        
        # dZ = dA * d(ReLU)/dZ
        dZ = np.array(dA, copy=True)
        dZ[Z <= 0] = 0
        return dZ

    def _binary_cross_entropy(self, y: ArrayLike, y_hat: ArrayLike) -> float:
        """
        Binary cross entropy loss function.

        Args:
            y_hat: ArrayLike
                Predicted output.
            y: ArrayLike
                Ground truth output.

        Returns:
            loss: float
                Average loss over mini-batch.
        """
        # epsilon to prevent log(0)
        epsilon = 1e-12
        y_hat = np.clip(y_hat, epsilon, 1 - epsilon)

        
        m = y.shape[1] # number of samples in the batch 

        # L = -(1/m) * sum[ y * log(y_hat) + (1 - y) * log(1 - y_hat) ]
        loss = -np.sum(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat)) / m
        return float(loss)

    def _binary_cross_entropy_backprop(self, y: ArrayLike, y_hat: ArrayLike) -> ArrayLike:
        """
        Binary cross entropy loss function derivative for backprop.

        Args:
            y_hat: ArrayLike
                Predicted output.
            y: ArrayLike
                Ground truth output.

        Returns:
            dA: ArrayLike
                partial derivative of loss with respect to A matrix.
        """
        # clip y_hat to avoid division by zero 
        epsilon = 1e-12
        y_hat = np.clip(y_hat, epsilon, 1 - epsilon)

        # dL/dA = -(y / y_hat) + (1 - y) / (1 - y_hat)
        return -(y / y_hat) + ((1 - y) / (1 - y_hat))

    def _mean_squared_error(self, y: ArrayLike, y_hat: ArrayLike) -> float:
        """
        Mean squared error loss.

        Args:
            y: ArrayLike
                Ground truth output.
            y_hat: ArrayLike
                Predicted output.

        Returns:
            loss: float
                Average loss of mini-batch.
        """
        
        m = y.shape[1] # m = number of samples in the batch

        # L = (1/m) * sum[ (y - y_hat)^2 ]
        loss = np.sum((y - y_hat) ** 2) / m
        return float(loss)

    def _mean_squared_error_backprop(self, y: ArrayLike, y_hat: ArrayLike) -> ArrayLike:
        """
        Mean square error loss derivative for backprop.

        Args:
            y_hat: ArrayLike
                Predicted output.
            y: ArrayLike
                Ground truth output.

        Returns:
            dA: ArrayLike
                partial derivative of loss with respect to A matrix.
        """
        # dL/dA = (2/m) * (y_hat - y)
        return 2 * (y_hat - y) / y.shape[1]