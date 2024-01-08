import pickle
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch_model
from random import shuffle
import torch.nn as nn

from tqdm import tqdm # progress bar

# Load data

data_original = np.loadtxt("training_data.csv", ndmin=2)    # Load data from file. Order is [theta1, theta2, x, y]
data_original = np.reshape(data_original,(-1,4))            # Reshape data to 4 columns since original data is all in 1 column

data_before = data_original.copy()                          # Save a copy of the original data for later use

# Standardize data

# data_original = data_original - np.mean(data_original, axis=0) # Subtract mean from data
# data_original = data_original/np.std(data_original, axis=0)    # Divide by standard deviation

np.random.shuffle(data_original)                            # Shuffle the data in order to get red of any biases in the way data was collected

train_set = data_original[:int(0.8*data_original.shape[0]),:]   # Split the data into a training set and a test set. Don't think its actually neeeded
test_set = data_original[int(0.8*data_original.shape[0]):,:]

angledata = train_set[:,0:2]                                # Split the data into the angle data and the xy data
xydata = train_set[:,2:4]



# Use GPU?
device = 'cpu'
if torch.cuda.is_available():
    print("Using GPU")
    device = 'cuda'
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

print(device)

x = torch.from_numpy(xydata).float()                        # Convert the data to tensors
y = torch.from_numpy(angledata).float()

#print(x)

if device == 'cuda':
    x = x.cuda()
    y = y.cuda()

# Define neural network
    
h = 32                                                     # Number of neurons in each hidden layer

model = torch_model.MLPNet(2, h, 2)                       # If you want to use a MLP

# model = nn.Sequential(nn.Flatten(),                         # Better performance reached with this model
#                      nn.Linear(2,h),
#                      nn.ReLU(),
#                      nn.Linear(h,h),
#                      nn.ReLU(),
#                      nn.Linear(h,h),
#                      nn.ReLU(),
#                      nn.Linear(h,2))

print(model)

lr = 0.001                                                  # Choose Learning rate                                       

optimizer = torch.optim.Adam(model.parameters(), lr=lr)
loss_func = torch.nn.MSELoss()
num_epochs = 200000                                        # Choose number of epochs                    

g = 0.999
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=g)

l_vec = np.zeros(num_epochs)

for t in tqdm(range(num_epochs)):
    # TODO Train network
    prediction = model(x) # Forward pass prediction. Saves intermediary values required for backwards pass
    loss = loss_func(prediction, y) # Computes the loss for each example, using the loss function defined above
    optimizer.zero_grad() # Clears gradients from previous iteration
    loss.backward() # Backpropagation of errors through the network
    optimizer.step() # Updating weights
    scheduler.step()

    l = loss.data
    if device == 'cuda':
        l = l.cpu()
    #print(l.numpy())
    l_vec[t] = l.numpy()

plt.plot(l_vec)
plt.legend(['Training loss'])
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.yscale('log')

angledata_test = test_set[:,0:2]                                # Split the data into the angle data and the xy data
xydata_test = test_set[:,2:4]
print("Angle data test: ", angledata_test)

x_test = torch.from_numpy(xydata_test).float()
y_test = torch.from_numpy(angledata_test).float()                        # Convert the data to tensors

prediction_test = model(x_test)
print(prediction_test)
loss_test = loss_func(prediction_test, y_test)
l_test = loss_test.data
print(l_test.numpy())

# scaled_up = prediction_test.detach().numpy()*np.std(data_before[:,0:2], axis=0) + np.mean(data_before[:,0:2], axis=0)
# loss_test_scaled = loss_func(scaled_up, torch.from_numpy(angledata_test).float()).data
# print(loss_test_scaled.numpy())

np.savetxt("loss_plot.csv", l_vec, delimiter=",")           # Save the loss values to a file

torch.save(model.state_dict(), 'trained_model.pth')
plt.show()


## Parameter TIPS - Try
# Adam with lr = 0.001
# StepLR scheduler with step_size=100, and gamma = 0.999
# Two hidden layers with 18 units each