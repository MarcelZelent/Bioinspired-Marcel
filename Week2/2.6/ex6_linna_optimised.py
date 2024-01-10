import numpy as np
import matplotlib.pyplot as plt

from robot import SingleLink
from cmac2 import CMAC

## Initialize simulation
Ts = 1e-3
T_end = 10 # in one trial
n_steps = int(T_end/Ts) # in one trial
n_trials = 50

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

## TODO: CMAC initialization
n_rfs = 11

xmin = [-np.pi,-np.pi]
xmax = [np.pi, np.pi]

c = CMAC(n_rfs, xmin, xmax, 1e-6)

flag = 0
i_low= 0

epoch = 0.00
e_vec = []
mean_before = 0
mean_vec = []
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
    # print("cmac",tau_cmac)
    tau = tau_m #+ tau_cmac
    # print("taum",tau_m)
    # print("tau",tau)
    # Iterate simulation dynamics
    c.learn(tau_m)
    plant.step(tau)
    
    err = error**2
    mean_now = mean_before + (err - mean_before)/(i+1)
    mean_before = mean_now
    mean_vec.append(mean_now)
    
    # print("e_vec",mse)
    
  
    epoch = epoch +1
    # # print(epoch)
    # if (flag == 0 and epoch > 100):
    #     if mse<= 0.1:
    #         print("epoch",epoch)
    #         flag = 1
    #         i_low = i
    #         # T = 2
    
    # if flag == 1:
    #     if (i>i_low+n_steps):
    #         print(T)
    #         print(mse)
    #         break

    if i%10000 == 0:
        print(i, mean_now)
            
                
            
        
    
    theta_vec[i] = plant.theta
    theta_ref_vec[i] = theta_ref

    # for _ in range(1000):
    #     e_vec = []
    #     for x1 in np.linspace(0, 1, 11):
    #         for x2 in np.linspace(0, 1, 11):
    #             x = [x1, x2]

    #             yhat = c.predict(x)

    #             yd = np.arctan2(x[0], x[1])

    #             e = yd - yhat

    #             c.learn(e)
    #             e_vec.append(e**2)

        # print(np.mean(e_vec))

## Plotting

print("mean",mean_vec[-1])

plt.plot(t_vec, theta_vec, label='theta')
plt.plot(t_vec, theta_ref_vec, '--', label='reference')
plt.plot(t_vec, mean_vec, label='error')
plt.legend()

## Plot trial error
#error_vec = theta_ref_vec - theta_vec
#l = int(T/Ts)
#trial_error = np.zeros(n_trials)
#for t in range(n_trials):
#    trial_error[t] = np.sqrt( np.mean( error_vec[t*l:(t+1)*l]**2 ) )
#plt.figure()
#plt.plot(trial_error)

plt.show()