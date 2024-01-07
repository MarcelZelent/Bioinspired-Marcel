import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def linear_activation(x):
    return x

def initialize_parameters(input_size, hidden_size, output_size):
    np.random.seed(42)
    weights_hidden = np.random.rand(hidden_size, input_size)
    biases_hidden = np.zeros((hidden_size, 1))

    weights_output = np.random.rand(output_size, hidden_size)
    biases_output = np.zeros((output_size, 1))

    parameters = {
        'weights_hidden': weights_hidden,
        'biases_hidden': biases_hidden,
        'weights_output': weights_output,
        'biases_output': biases_output
    }

    return parameters

def forward_propagation(inputs, parameters):
    # Hidden layer
    hidden_activation = sigmoid(np.dot(parameters['weights_hidden'], inputs) + parameters['biases_hidden'])

    # Output layer
    output_activation = linear_activation(np.dot(parameters['weights_output'], hidden_activation) + parameters['biases_output'])

    cache = {
        'hidden_activation': hidden_activation,
        'output_activation': output_activation
    }

    return output_activation, cache

def backward_propagation(inputs, outputs, cache, parameters, learning_rate):
    m = inputs.shape[1]

    # Output layer
    output_error = outputs - cache['output_activation']
    output_delta = output_error  # Linear activation function derivative is 1

    # Hidden layer
    hidden_error = np.dot(parameters['weights_output'].T, output_delta)
    hidden_delta = cache['hidden_activation'] * (1 - cache['hidden_activation']) * hidden_error

    # Update parameters
    parameters['weights_output'] += learning_rate * np.dot(output_delta, cache['hidden_activation'].T) / m
    parameters['biases_output'] += learning_rate * np.sum(output_delta, axis=1, keepdims=True) / m

    parameters['weights_hidden'] += learning_rate * np.dot(hidden_delta, inputs.T) / m
    parameters['biases_hidden'] += learning_rate * np.sum(hidden_delta, axis=1, keepdims=True) / m

    return parameters

def train_neural_network(inputs, outputs, hidden_size, learning_rate, epochs):
    input_size = inputs.shape[0]
    output_size = outputs.shape[0]

    parameters = initialize_parameters(input_size, hidden_size, output_size)

    for epoch in range(epochs):
        output_activation, cache = forward_propagation(inputs, parameters)
        parameters = backward_propagation(inputs, outputs, cache, parameters, learning_rate)

        if epoch % 1000 == 0:
            cost = np.mean(np.square(outputs - output_activation)) / 2
            print(f'Epoch {epoch}, Cost: {cost}')

    return parameters

# Example usage:
# Assuming 'inputs' and 'outputs' are your input and output data, with each column representing a data point
inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]]).T
outputs = np.array([[0, 1, 1, 0]])

hidden_size = 2
learning_rate = 0.1
epochs = 10000

trained_parameters = train_neural_network(inputs, outputs, hidden_size, learning_rate, epochs)

# Test the trained neural network
output_activation, _ = forward_propagation(inputs, trained_parameters)
print(f'Output: {output_activation}')
print("Final Weights: ", trained_parameters['weights_hidden'], trained_parameters['weights_output'])
