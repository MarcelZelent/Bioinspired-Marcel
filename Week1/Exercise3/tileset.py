import numpy as np
import matplotlib.pyplot as plt

n_t1 = 20
n_t2 = 20

t1 = np.tile(np.linspace(-85, 86, n_t1), n_t2) # repeat the vector
t2 = np.repeat(np.linspace(0, 86, n_t2), n_t1) # repeat each element
thetas = np.stack((t1,t2))

num_datapoints = n_t1*n_t2

# Plotting
plt.figure(figsize=(6, 6))
plt.scatter(thetas[0], thetas[1])
plt.xlabel('t1')
plt.ylabel('t2')
plt.title('Grid of Points')
plt.grid(True)
plt.show()