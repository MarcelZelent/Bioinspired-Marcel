import numpy as np
import matplotlib.pyplot as plt

# Point 1.1
def LIF(um_0, I, T):
    # Parameters:
    # um_0 : initial membrane voltage
    # I : input current
    # T : maximum simulation time
    Rm = 10*1e6 # Mega Ohm
    Cm = 1*1e-9 # Nano Farad
    u_thresh = -50*1e-3 # milli Volt
    u_rest = -65*1e-3 # milli Volt
    
    tau_m = Rm * Cm
    
    delta_t = 1e-5
    um_t = np.zeros((int(T//delta_t)))
    um_t[0] = um_0
    
    dum_dt = lambda um_t: (u_rest - um_t + Rm*I)/tau_m
    
    # TODO Calculate the um_t
    
    for t in range(len(um_t)-1):
        if um_t[t] > u_thresh:
            um_t[t+1] = u_rest
        else:
            um_t[t+1] = um_t[t] + dum_dt(um_t[t]) * delta_t
    
    return um_t

# Point 1.2
# TODO calculate the membrane potential using the LIF function
membrane_potential = LIF(-75e-3, 1e-9, 100e-3)

plt.figure(figsize=(7,5))
plt.plot(list(range(int(0.1//1e-5))), membrane_potential)
plt.show()


# Point 1.3
# TODO define a function to calculate the interspike intervals
def calculate_isi(array, timestep, threshold):
    out = []
    for index, value in enumerate(array):
        if value > threshold:
            out.append(index * timestep)
    return out

# TODO define a function to calculate the spiking frequency of a whole experiment
def spiking_frequency(array):
    periods = [array[i+1] - array[i] for i in range(len(array)-1)]
    # average period
    if len(periods) == 0:
        freq = 0
    else:
        avg_prd = sum(periods)/len(periods)
        freq = 1 / avg_prd
    return freq
        


# Point 1.4
plt.figure(figsize=(7,5))
spikes = []
# TODO write the code to accumulate the spikes
for I in list(np.arange(0,5.5e-9, 0.5e-9)):
    spikes.append(spiking_frequency(calculate_isi(
        LIF(-75e-3, I, 100e-3), 1e-5, -50e-3)))
plt.plot(list(np.arange(0,5.5e-9, 0.5e-9)), spikes)
plt.xlabel('Constant current')
plt.ylabel('Spiking frequency')
plt.show()