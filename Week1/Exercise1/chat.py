import numpy as np

class Perceptron:
    def __init__(self, num_inputs, learning_rate=0.01):
        self.weights = np.random.rand(num_inputs)
        self.bias = np.random.rand()
        self.learning_rate = learning_rate

    def predict(self, inputs):
        weighted_sum = np.dot(inputs, self.weights) + self.bias
        return 1 if weighted_sum > 0 else 0

    def train(self, training_data, epochs=100):
        for epoch in range(epochs):
            for inputs, target in training_data:
                prediction = self.predict(inputs)
                error = target - prediction

                # Update weights and bias
                self.weights += self.learning_rate * error * inputs
                self.bias += self.learning_rate * error

# Example usage:

# Define training data (inputs and corresponding targets)
training_data = [
    (np.array([0, 0]), 0),
    (np.array([0, 1]), 1),
    (np.array([1, 0]), 1),
    (np.array([1, 1]), 1),
]

# Create a perceptron with 2 inputs
perceptron = Perceptron(num_inputs=2)

# Train the perceptron
perceptron.train(training_data)

# Test the trained perceptron
test_inputs = np.array([1, 0])
prediction = perceptron.predict(test_inputs)

print(f"Prediction for {test_inputs}: {prediction}")
