import numpy as np
import matplotlib.pyplot as plt

from adaptive_filter.cerebellum import AdaptiveFilterCerebellum
from robot import SingleLink

Ts = 1e-3
n_inputs = 1
n_outputs = 1
n_bases = 60
beta = 1e-8

c = AdaptiveFilterCerebellum(Ts, n_inputs, n_outputs, n_bases, beta)

## Initialize simulation
T_end = 10 # in one trial
n_steps = int(T_end/Ts) # in one trial
n_trials = 5

plant = SingleLink(Ts)

## Logging variables
t_vec = np.array([Ts*i for i in range(n_steps*n_trials)])

theta_vec = np.zeros(n_steps*n_trials)
theta_ref_vec = np.zeros(n_steps*n_trials)

## Feedback controller variables
Kp = 20
Kv = 3

## TODO: Define parameters for periodic reference trajectory
A = np.pi
T = 5

flag = 0
i_low= 0
af_out = 0

epoch = 0.00
e_vec = np.zeros(n_steps*n_trials)
af_out_vec = []
e_vec_sq = np.zeros(n_steps*n_trials)

mean_before = 0
mean_vec = []

rate = 10000


## Simulation loop
for i in range(n_steps*n_trials):
    t = i*Ts
    theta_ref = A * np.sin(2*np.pi * t/T)

    # Measure
    theta = plant.theta
    omega = plant.omega

    # print(theta)

    theta_vec[i] = float(theta)
    theta_ref_vec[i] = theta_ref

    # Feedback controler
    error = (theta_ref - theta)
    err_fb = error + af_out

    u = Kp * err_fb + Kv* (-omega)
    
    af_out = c.step(u, error)

    # Iterate simulation dynamics

    plant.step(u)
    
    # e_vec_sq[i] = np.square(error)
    # e_vec[i] = np.sum(e_vec_sq[0:i+1])/(i+1)

    err = error**2
    mean_now = mean_before + (err - mean_before)/(i+1)
    mean_before = mean_now
    mean_vec.append(mean_now)

    if i%rate == 0:
        print(i/rate, mean_now)


    epoch = epoch +1

## Plotting
    
plt.plot(t_vec, theta_vec, label='theta')
plt.plot(t_vec, theta_ref_vec, '--', label='reference')
plt.plot(t_vec, mean_vec, label='error')
plt.ylim([-A*2, A*2])
plt.legend()
plt.show()
