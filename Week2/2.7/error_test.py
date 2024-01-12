import numpy as np

error = 1000
error_vec = np.zeros(1000)

for i in range(500):
    # print(error)
    error_vec[i] = error
    if i % 100 == 0:
        print(np.mean(error))
    error = error-1
