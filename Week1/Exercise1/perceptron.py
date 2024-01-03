import numpy as np
from activation import ActivationFunction
import matplotlib.pyplot as plt

class SignActivation(ActivationFunction):
   """ 
         Sign activation: `f(x) = 1 if x > 0, 0 if x <= 0`
   """
   def forward(self, x):
      """
         This is the output function.
         TODO: Define the correct return function, given input `x`
      """
      if x>0:
         return 1
      else:
         return 0
      
   def gradient(self, x):
      """
            Function derivative.
            Define the correct return value (derivative), given input `x`
      """
      return None

class Perceptron:
   """ 
      Perceptron neuron model
      Parameters
      ----------
      n_inputs : int
         Number of inputs
      act_f : Subclass of `ActivationFunction`
         Activation function
   """
   def __init__(self, n_inputs, act_f):
      """
         Perceptron class initialization
         TODO: Write the code to initialize weights and save the given activation function
      """
      if not isinstance(act_f, type) or not issubclass(act_f, ActivationFunction):
         raise TypeError('act_f has to be a subclass of ActivationFunction (not a class instance).')
      # weights
      mean = 0
      standard_deviation = 1
      self.w = np.random.normal(mean, standard_deviation, n_inputs)
      self.w_bias = np.random.normal(mean, standard_deviation, 1)
      # activation function
      self.f = act_f()

      if self.f is not None and not isinstance(self.f, ActivationFunction):
         raise TypeError("self.f should be a class instance.")

   def activation(self, x):
      """
         It computes the activation `a` given an input `x`
         TODO: Fill in the function to provide the correct output
         NB: Remember the bias
      """
      a = np.dot(self.w, x) + self.w_bias
      return a

   def output(self, a):
      """
         It computes the neuron output `y`, given the activation `a`
         TODO: Fill in the function to provide the correct output
      """
      y = self.f.forward(a)
      return y

   def predict(self, x):
      """
         It computes the neuron output `y`, given the input `x`
         TODO: Fill in the function to provide the correct output
      """
      prediction = self.output(self.activation(x))
      return prediction

   def gradient(self, a):
      """
         It computes the gradient of the activation function, given the activation `a`
      """
      return self.f.gradient(a)

if __name__ == '__main__':
   data = np.array( [ [0.5, 0.5, 0], [1.0, 0, 0], [2.0, 3.0, 0], [0, 1.0, 1], [0, 2.0, 1], [1.0, 2.2, 1] ] )
   xdata = data[:,:2]
   ydata = data[:,2]
   print(xdata)
   print(ydata)
   
## TODO Test your activation function
a = SignActivation()
print(a.forward(2))
print(a.forward(0))

## TODO Test perceptron initialization
p = Perceptron(2, SignActivation)
print(p.predict(xdata[0,:]) )

## TODO Learn the weights
r = 0.1 # learning rate
## calculate the error and update the weights
epochs = 1000
for i in range(epochs):
   count = 0
   for j in xdata:
      a = p.activation(j)
      y = p.output(a)
      error = ydata[count] - y
      p.w = p.w + r*error*j
      p.w_bias = p.w_bias + r*error
      count += 1
print(p.w)
## TODO plot points and linear decision boundary

# PLOT THE xdata POINTS
plt.plot(xdata[:,0],xdata[:,1],'bo')

# PLOT THE DECISION BOUNDARY
xp = np.linspace(0,2,100)
yp = -p.w[0]/p.w[1]*xp - p.w_bias/p.w[1]

plt.plot(xp,yp, 'k--')
plt.xlabel('x1')
plt.ylabel('x2')
plt.xlim([-1, 4])
plt.ylim([-1, 4])
plt.show()



