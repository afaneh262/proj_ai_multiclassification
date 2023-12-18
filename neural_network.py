"""
@author: Wajed Afaneh
"""
import numpy as np
from sklearn.preprocessing import OneHotEncoder

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    sigmoid_x = sigmoid(x)
    return sigmoid_x * (1 - sigmoid_x)


def tanh(x):
    return np.tanh(x)

def tanh_derivative(x):
    tanh_x = tanh(x)
    return 1 - tanh_x**2


def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return np.where(x > 0, 1, 0)


def leaky_relu(x, alpha=0.01):
    return np.maximum(alpha * x, x)

def leaky_relu_derivative(x, alpha=0.01):
    return np.where(x > 0, 1, alpha)


def elu(x, alpha=1.0):
    return np.where(x > 0, x, alpha * (np.exp(x) - 1))

def elu_derivative(x, alpha=1.0):
    return np.where(x > 0, 1, alpha * np.exp(x))


def softmax(x):
    exp_values = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_values / np.sum(exp_values, axis=-1, keepdims=True)


def linear(x):
    return x

def linear_derivative(x):
    return 1


class NeuralNetwork:
    def __init__(self, hidden_size, activation_function='Sigmoid', output_function='Softmax', epochs=100,
                 learning_rate=0.01, goal=None, progress_cal=None):
        self.predictions = None
        self.hidden_output = None
        self.output_input = None
        self.bias_output = None
        self.weights_hidden_output = None
        self.bias_hidden = None
        self.weights_input_hidden = None
        self.output_size = None
        self.hidden_input = None
        self.input_size = None
        self.hidden_size = hidden_size
        self.activation_function = activation_function
        self.output_function = output_function
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.goal = goal
        self.progress_cal = progress_cal

    def cal_function_output(self, function_name, hidden_input):
        if function_name == 'Sigmoid':
            return sigmoid(hidden_input)
        elif function_name == 'Tanh':
            return tanh(hidden_input)
        elif function_name == 'ReLU':
            return relu(self.output_input)
        elif function_name == 'Leaky ReLU':
            return leaky_relu(self.output_input)
        elif function_name == 'ELU':
            return elu(self.output_input)
        elif function_name == 'Softmax':
            return softmax(self.output_input)
        elif function_name == 'Linear':
            return linear(self.output_input)
        else:
            raise ValueError("Invalid activation function")
    
    def cal_derivative_function_output(self, function_name, hidden_input):
        if function_name == 'Sigmoid':
            return sigmoid_derivative(hidden_input)
        elif function_name == 'Tanh':
            return tanh_derivative(hidden_input)
        elif function_name == 'ReLU':
            return relu_derivative(hidden_input)
        elif function_name == 'Leaky ReLU':
            return leaky_relu_derivative(hidden_input)
        elif function_name == 'ELU':
            return elu_derivative(hidden_input)
        elif function_name == 'Linear':
            return linear_derivative(hidden_input)
        else:
            raise ValueError("Invalid activation function")
        
    def forward(self, X):
        # Input to hidden layer
        self.hidden_input = np.dot(X, self.weights_input_hidden) + self.bias_hidden

        # Apply activation function to hidden layer
        self.hidden_output = self.cal_function_output(self.activation_function, self.hidden_input)

        # Hidden to output layer
        self.output_input = np.dot(self.hidden_output, self.weights_hidden_output) + self.bias_output

        # Initialize predictions before softmax
        self.predictions = self.output_input

        # Apply activation to output layer
        self.predictions = self.cal_function_output(self.output_function, self.output_input)

    def backward(self, X, y, learning_rate):
        # Calculate loss
        m = X.shape[0]
        loss = -np.sum(np.log(self.predictions[range(m), np.argmax(y, axis=1)])) / m

        # Backpropagation
        output_error = self.predictions - y
        output_error /= m

        # Update weights and biases for the output layer
        d_weights_hidden_output = np.dot(self.hidden_output.T, output_error)
        d_bias_output = np.sum(output_error, axis=0, keepdims=True)

        # Update weights and biases for the hidden layer
        hidden_error = np.dot(output_error, self.weights_hidden_output.T)
        hidden_error *= self.cal_derivative_function_output(self.activation_function, self.hidden_output)


        d_weights_input_hidden = np.dot(X.T, hidden_error)
        d_bias_hidden = np.sum(hidden_error, axis=0, keepdims=True)

        # Update weights and biases using gradient descent
        self.weights_input_hidden -= learning_rate * d_weights_input_hidden
        self.bias_hidden -= learning_rate * d_bias_hidden
        self.weights_hidden_output -= learning_rate * d_weights_hidden_output
        self.bias_output -= learning_rate * d_bias_output

        return loss

    def train(self, X, y):
        # Determine input_size and output_size from the dataset
        self.input_size = X.shape[1]
        self.output_size = np.unique(y).size

        # One-hot encode the target labels for multiclass classification
        if self.output_size > 2:
            encoder = OneHotEncoder(sparse_output=False)
            y_one_hot = encoder.fit_transform(y.reshape(-1, 1))
        else:
            # For binary classification, y can remain as is
            y_one_hot = y.reshape(-1, 1)

        # Initialize weights and biases
        self.weights_input_hidden = np.random.rand(self.input_size, self.hidden_size)
        self.bias_hidden = np.zeros((1, self.hidden_size))
        self.weights_hidden_output = np.random.rand(self.hidden_size, self.output_size)
        self.bias_output = np.zeros((1, self.output_size))

        for epoch in range(self.epochs):
            # Forward pass
            self.forward(X)

            # Backward pass and optimization
            loss = self.backward(X, y_one_hot, self.learning_rate)

            # Print Epoch
            if(self.progress_cal):
                self.progress_cal(epoch+1)

            # Check goal accuracy
            if self.goal is not None:
                predictions = self.predict(X)
                accuracy = np.mean(predictions == y)

                if accuracy >= self.goal:
                    break

    def predict(self, X):
        self.forward(X)
        return np.argmax(self.predictions, axis=1)
