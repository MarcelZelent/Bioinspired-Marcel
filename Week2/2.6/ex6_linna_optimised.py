import numpy as np
import matplotlib.pyplot as plt

from robot import SingleLink
from cmac2 import CMAC

## Initialize simulation
Ts = 1e-3
T_end = 10 # in one trial
n_steps = int(T_end/Ts) # in one trial
n_trials = 30

plant = SingleLink(Ts)

## Logging variables
t_vec = np.array([Ts*i for i in range(n_steps*n_trials)])

theta_vec = np.zeros(n_steps*n_trials)
theta_ref_vec = np.zeros(n_steps*n_trials)

## Feedback controller variables
Kp = 10
Kv = 0


## TODO: Define parameters for periodic reference trajectory
A = np.pi
T = 5
t = Ts
theta_ref = A * np.sin(2*np.pi * t/T)

## TODO: CMAC initialization
n_rfs = 11

xmin = [-np.pi,-np.pi]
xmax = [np.pi, np.pi]

c = CMAC(n_rfs, xmin, xmax, 0.01)

flag = 0
i_low= 0

epoch = 0.00
e_vec = []
mean_before = 0
mean_vec = []

rate = 10000

## Simulation loop
for i in range(n_steps*n_trials):
    t = i*Ts
    ## TODO: Calculate the reference at this time step
    # theta_ref = np.pi/4
    theta_ref = A * np.sin(2*np.pi * t/T)
    
    
    # Measure
    theta = plant.theta
    omega = plant.omega

    # Feedback controler
    error = (theta_ref - theta)
    tau_m = Kp * error + Kv* (-omega)
    
    ## TODO: Implement the CMAC controller into the loop
    
    x = [theta_ref,theta]
    tau_cmac = c. predict(x)
 
    tau = tau_m + tau_cmac

    # Iterate simulation dynamics
    c.learn(tau_m)
    plant.step(tau)
    
    err = error**2
    mean_now = mean_before + (err - mean_before)/(i+1)
    mean_before = mean_now
    mean_vec.append(mean_now)
    
  
    epoch = epoch +1

    if i%rate == 0:
        print(i/rate, mean_now)    
    
    theta_vec[i] = plant.theta
    theta_ref_vec[i] = theta_ref


## Plotting

print("mean",mean_vec[-1])

plt.plot(t_vec, theta_vec, label='theta')
plt.plot(t_vec, theta_ref_vec, '--', label='reference')
plt.plot(t_vec, mean_vec, label='error')
plt.legend()

plt.show()