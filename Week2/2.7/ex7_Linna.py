import numpy as np
import matplotlib.pyplot as plt

from adaptive_filter.cerebellum import AdaptiveFilterCerebellum
from robot import SingleLink

Ts = 1e-3
n_inputs = 1
n_outputs = 1
n_bases = 100
beta =10e-8

#c = AdaptiveFilterCerebellum(Ts, n_inputs, n_outputs, n_bases, beta)

## TODO: Paste your experiment code from exercise 2.6
## Initialize simulation
# Ts = 1e-3
T_end = 10 # in one trial
n_steps = int(T_end/Ts) # in one trial
n_trials = 1

plant = SingleLink(Ts)

## Logging variables
t_vec = np.array([Ts*i for i in range(n_steps*n_trials)])

theta_vec = np.zeros(n_steps*n_trials)
theta_ref_vec = np.zeros(n_steps*n_trials)

## Feedback controller variables
# Kp = 150
# Kv = 3
# Kp = 10
# Kv = 1
Kp=30
Kv=1

## TODO: Define parameters for periodic reference trajectory
A = np.pi
T = 5
t = Ts
theta_ref = A * np.sin(2*np.pi * t/T)

## TODO: CMAC initialization


c = AdaptiveFilterCerebellum(Ts,n_inputs, n_outputs,n_bases,beta)


epoch = 0.00
e_vec = []

## Simulation loop
for i in range(n_steps*n_trials):
    t = i*Ts
    ## TODO: Calculate the reference at this time step
    # theta_ref = np.pi/4
    theta_ref = A * np.sin(2*np.pi * t/T)
    # c = AdaptiveFilterCerebellum(Ts,n_inputs, n_outputs,n_bases,beta)
    
    # Measure
    theta = plant.theta
    omega = plant.omega


    # Feedback controler
    error = (theta_ref - theta)
    C = c.step(u, error)
    # print(C)
    error = error +C
    tau_m = Kp * error + Kv* (-omega)
    u = tau_m
    
    plant.step(u)
  
    epoch = epoch +1
    # print(epoch)
    # if (flag == 0 and epoch > 100):
    #     if mse<= 0.1:
    #         print("epoch",epoch)
    #         flag = 1
    #         i_low = i
             
    
    theta_vec[i] = plant.theta
    theta_ref_vec[i] = theta_ref


## TODO: Change the code to the recurrent architecture
# You can update the cerebellum with: C = c.step(u, error)

## TODO: Plot results
plt.plot(t_vec, theta_vec, label='theta')
plt.plot(t_vec, theta_ref_vec, '--', label='reference')
plt.legend()
plt.show()

