import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

# Specify the actual folder and filenames
folder_path = 'C:/Users/qyu38/Code/RNN_prof_est/npy_all_five'  # Replace with your actual folder path
# folder_path_test = 'C:/Users/qyu38/Code/RNN_prof_est/npy'
# X_filename = '/X_standardized.npy' % this is the simulated data
# y_filename = '/y.npy' % this is the simulated data

X_filename_train = '/X_all_route_aligned_2000_seq_100_train_hatch_only_prof_vel.npy'
y_filename_train = '/y_all_route_aligned_2000_seq_100_train_hatch_only_prof_vel.npy'

X_filename_val = '/X_all_route_aligned_2000_seq_100_validate_hatch_only_prof_vel.npy'
y_filename_val = '/y_all_route_aligned_2000_seq_100_validate_hatch_only_prof_vel.npy'

X_filename_test = '/X_all_route_aligned_2000_seq_100_test_hatch_only_prof_vel.npy'
y_filename_test = '/y_all_route_aligned_2000_seq_100_test_hatch_only_prof_vel.npy'


model_save_path = 'C:/Users/qyu38/Code/RNN_prof_est/model/model_all_route_real_seq_100_FCL_hatch_only_LSTM_prof_vel_lr0d001_epoch_200.pth'


X_standardized_train = np.load(folder_path + X_filename_train)
# breakpoint()
X_train = np.delete(X_standardized_train, 0, axis=2) #to leave out the simulated response
# X_train = np.delete(X_standardized_train, -1, axis=2) #to leave out the real response, train using simulated

X_standardized_val = np.load(folder_path + X_filename_val)
X_val = np.delete(X_standardized_val, 0, axis=2) #to leave out the simulated response
# X_val = np.delete(X_standardized_val, -1, axis=2) #to leave out the real response, train using simulated

X_standardized_test = np.load(folder_path + X_filename_test)
X_test = np.delete(X_standardized_test, 0, axis=2) #to leave out the simulated response
# X_test = np.delete(X_standardized_test, -1, axis=2) #to leave out the real response, testing using simulated

##########do the below if the model is trained on simulated data and tested on real data########
# X_test_real = np.delete(X_standardized_test, 5, axis=2) #to leave out the simulated response
# col1 = 5  # Second last column
# col2 = 6  # Last column
# # Perform the swap
# X_test_real[:, :, [col1, col2]] = X_test_real[:, :, [col2, col1]] 
################################################################################################

y_train = np.load(folder_path + y_filename_train)
y_val = np.load(folder_path + y_filename_val)
y_test = np.load(folder_path + y_filename_test)

# LSTM model
class LSTMNet(nn.Module):
    def __init__(self):
        super(LSTMNet, self).__init__()
        self.lstm = nn.LSTM(input_size=2, hidden_size=100, num_layers=4, batch_first=True)
        self.fc_layers = nn.Sequential(
            nn.Tanh(),
            nn.Linear(100, 100),
            nn.Tanh(),
            nn.Linear(100, 100),
            nn.Tanh(),
            nn.Linear(100, 100),
            nn.Tanh(),
            nn.Linear(100, 1)
        )

    def forward(self, x):
        x, _ = self.lstm(x)  # LSTM output is (output, (h_n, c_n)), we only need the output
        x = x[:, -1, :]  # Get the last time step output from LSTM
        out = self.fc_layers(x)
        return out

net = LSTMNet()

# class FCLNet(nn.Module):
#     def __init__(self):
#         super(FCLNet, self).__init__()
#         self.fc_layers = nn.Sequential(
#             nn.Tanh(),
#             nn.Linear(2, 100),
#             nn.Tanh(),
#             nn.Linear(100, 100),
#             nn.Tanh(),
#             nn.Linear(100, 100),
#             nn.Tanh(),
#             nn.Linear(100, 1)  # Final layer outputs a single value per sample
#         )

#     def forward(self, x):
#         out = self.fc_layers(x)
#         out = out.view(out.size(0), -1)  # Flatten to [batch_size, 1]
#         # print(f"Model output shape: {out.shape}")  # Debugging statement
#         return out

# net = FCLNet()

# Set the device to GPU if available, otherwise CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
net.to(device)  # Move the model to the specified device

batch_size = 64  # You can adjust the batch size

# Convert to PyTorch tensors
train_data = TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).float())
val_data = TensorDataset(torch.from_numpy(X_val).float(), torch.from_numpy(y_val).float())
test_data = TensorDataset(torch.from_numpy(X_test).float(), torch.from_numpy(y_test).float())
# test_data_real = TensorDataset(torch.from_numpy(X_test_real).float(), torch.from_numpy(y_test).float())

# Create DataLoaders
train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
val_loader = DataLoader(val_data, batch_size=batch_size)
test_loader = DataLoader(test_data, batch_size=batch_size)
# test_loader_real = DataLoader(test_data_real, batch_size=batch_size)

criterion = nn.MSELoss()  # Mean Squared Error Loss
optimizer = optim.Adam(net.parameters(), lr=0.0001)  # Using the Adam optimizer

# Set the number of epochs
num_epochs = 200  # You can adjust this
# print(len(train_loader))
print(f"Training loader length is: {len(train_loader)}")

for epoch in range(num_epochs):
    net.train()  # Set the model to training mode
    running_loss = 0.0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)  # Move data to the device
        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = net(inputs)
        # print(f"Labels shape: {labels.shape}")  # Debugging statement
        # print(f"Outputs shape: {outputs.shape}")  # Debugging statement
        loss = criterion(outputs, labels)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    # Validation loss
    net.eval()  # Set the model to evaluation mode
    val_loss = 0.0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)  # Move data to device/and can be deleted
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

    print(f"Epoch {epoch+1}/{num_epochs}, Training Loss: {running_loss/len(train_loader)}, Validation Loss: {val_loss/len(val_loader)}")

# Save the model state dict
torch.save(net.state_dict(), model_save_path)
print(f"Model saved to {model_save_path}")


################################### Start the testing process ############################################################################################
net.eval()  # Set the model to evaluation mode
test_loss = 0.0
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)  # Move data to the device
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        test_loss += loss.item()

print(f"Test Loss: {test_loss/len(test_loader)}")


predictions = []
ground_truth = []

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs = inputs.to(device)  # Move inputs to the same device as the model
        outputs = net(inputs)
        predictions.extend(outputs.cpu().numpy())
        ground_truth.extend(labels.cpu().numpy())

# Convert lists to arrays for plotting
predictions = np.array(predictions)
ground_truth = np.array(ground_truth)

mse = mean_squared_error(ground_truth, predictions)
print(f"mean_squared_error: {mse}")

# Plotting
plt.figure(figsize=(10,6))
plt.plot(ground_truth, label='Ground Truth')
plt.plot(predictions, label='Predicted')
plt.title('Comparison of Ground Truth and Predictions')
plt.xlabel('profile points')
plt.ylabel('roughness geometry (m)')
plt.legend()
########################################################################################################################################################################################
# Save the figure
# plt.savefig('C:/Users/qyu38/Code/RNN_prof_est/data/test_data_plot_real.png')  # Change the path and file name as needed
# import pickle
# # Save the figure
# with open('saved_figure.pkl', 'wb') as file:
#     pickle.dump(plt.gcf(), file)

plt.show()

breakpoint()
# import pandas as pd
# file_path = r"C:\Users\qyu38\Code\RNN_prof_est\performance_plot\predictions_sim_seq_100.csv"
# file_path_ground = r"C:\Users\qyu38\Code\RNN_prof_est\performance_plot\ground_truth_sim_seq_100.csv"
# df_ground = pd.DataFrame(ground_truth)
# df_predict = pd.DataFrame(predictions)

# df_predict.to_csv(file_path, index=False, header=False)
# df_ground.to_csv(file_path_ground, index=False, header=False)
# breakpoint()

