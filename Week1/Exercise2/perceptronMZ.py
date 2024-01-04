import numpy as np
from activationMZ import ActivationFunction
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
      self.w = #np.random.normal(mean, standard deviation, size)
      # activation function
      self.f =

      if self.f is not None and not isinstance(self.f, ActivationFunction):
         raise TypeError("self.f should be a class instance.")

   def activation(self, x):
      """
         It computes the activation `a` given an input `x`
         TODO: Fill in the function to provide the correct output
         NB: Remember the bias
      """
      a = 
      return a

   def output(self, a):
      """
         It computes the neuron output `y`, given the activation `a`
         TODO: Fill in the function to provide the correct output
      """
      y = 
      return y

   def predict(self, x):
      """
         It computes the neuron output `y`, given the input `x`
         TODO: Fill in the function to provide the correct output
      """
      return None

   def gradient(self, a):
      """
         It computes the gradient of the activation function, given the activation `a`
      """
      return self.f.gradient(a)

