import numpy as np
import matplotlib.pyplot as plt

from adaptive_filter.cerebellum import AdaptiveFilterCerebellum
from robot import SingleLink

Ts = 1e-3
n_inputs = 1
n_outputs = 1
n_bases = 50
beta = 1e-6

c = AdaptiveFilterCerebellum(Ts, n_inputs, n_outputs, n_bases, beta)

## TODO: Paste your experiment code from exercise 2.6
T_end = 10 # in one trial
n_steps = int(T_end/Ts) # in one trial
n_trials = 5

plant = SingleLink(Ts)

## Logging variables
t_vec = np.array([Ts*i for i in range(n_steps*n_trials)])

theta_vec = np.zeros(n_steps*n_trials)
theta_ref_vec = np.zeros(n_steps*n_trials)

## Feedback controller variables
# Kp = 150
# Kv = 3
Kp = 20
Kv = 3
# Kp=50
# Kv=1

## TODO: Define parameters for periodic reference trajectory
A = np.pi
T = 5
t = Ts
theta_ref = A * np.sin(2*np.pi * t/T)

e = np.zeros(n_steps*n_trials)
e_vec = np.zeros(n_steps*n_trials)
## Simulation loop
for i in range(n_steps*n_trials):
    t = i*Ts
    ## TODO: Calculate the reference at this time step
    # theta_ref = np.pi/4
    theta_ref = A * np.sin(2*np.pi * t/T)
    velocity_ref = (A*2*np.pi/T) * np.cos(2*np.pi * t/T)
    
    # Measure
    theta = plant.theta
    omega = plant.omega
    
    # Feedback controler
    error = (theta_ref - theta)
    if i == 0:
        e_fb = error
        u = Kp*e_fb+Kv*(-omega)
    C = c.step(u, error)
    e_fb = error + C
    u = Kp * e_fb + Kv* (-omega)
    # error = (theta_ref - theta)
    # u = Kp * error + Kv* (-omega)
    # print(error)
    # Iterate simulation dynamics
    plant.step(u)

    
    # err = error**2
    # e_vec.append(err[0])
    # print(e_vec)
    # mse = np.mean(e_vec)
    # print("e_vec",mse)
    e[i] = np.square(error)
    e_vec[i] = np.sum(e[0:i+1]) / (i + 1)
    
    theta_vec[i] = plant.theta
    theta_ref_vec[i] = theta_ref

## TODO: Change the code to the recurrent architecture
# You can update the cerebellum with: C = c.step(u, error)

## TODO: Plot results
plt.plot(t_vec, theta_vec, label='theta')
plt.plot(t_vec, theta_ref_vec, '--', label='reference')


plt.plot(t_vec, e_vec, label='error')
plt.legend()
# Plot trial error
# error_vec = theta_ref_vec - theta_vec
# l = int(T/Ts)
# trial_error = np.zeros(n_trials)
# for t in range(n_trials):
#    trial_error[t] = np.sqrt( np.mean( error_vec[t*l:(t+1)*l]**2 ) )
# plt.figure()
# plt.plot(trial_error)

plt.show()

