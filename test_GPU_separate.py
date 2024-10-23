import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

folder_path = 'C:/Users/qyu38/Code/RNN_prof_est/npy_all_five'  # Replace with your actual folder path
# X_filename_test = '/X_standardized_north_all_real_with_no_speedlow_test_aligned_2000_seq_100.npy'
# y_filename_test = '/y_north_all_real_with_no_speedlow_test_aligned_2000_seq_100.npy'

# X_standardized_test = np.load(folder_path + X_filename_test)
# X_test = np.delete(X_standardized_test, 7, axis=2) #to leave out the simulated response
# y_test = np.load(folder_path + y_filename_test)

# # LSTM model
# class LSTMNet(nn.Module):
#     def __init__(self):
#         super(LSTMNet, self).__init__()
#         self.lstm = nn.LSTM(input_size=7, hidden_size=100, num_layers=4, batch_first=True)#the first trial was 2 layers
#         self.fc = nn.Linear(100, 1)

#     def forward(self, x):
#         x, _ = self.lstm(x)
#         x = x[:, -1, :]  # Get the last time step
#         x = self.fc(x)
#         return x

# net = LSTMNet()

X_filename_test = '\X_all_route_aligned_2000_seq_100_test_hatch_only_prof.npy'
y_filename_test = '\y_all_route_aligned_2000_seq_100_test_hatch_only_prof.npy'


X_standardized_test = np.load(folder_path + X_filename_test)
# X_test = np.delete(X_standardized_test, 5, axis=2) #to leave out the simulated response
# X_test = np.delete(X_standardized_test, -1, axis=2) #to leave out the real response, testing using simulated
X_test = np.delete(X_standardized_test, 0, axis=2) #to leave out the simulated response
# X_test_swapped = X_test[..., ::-1]  # Swap the columns in the last dimension so the dimention is consistent with that trained using the sim

y_test = np.load(folder_path + y_filename_test)

# X_test = X_test_swapped.copy()
# y_test = y_test.copy()

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

# Set the device to GPU if available, otherwise CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load the trained model's state dictionary
# model_path = 'C:/Users/qyu38/Code/RNN_prof_est/model/model_all_route_sim_seq_100.pth'  # Change the file name if needed

# model_path=r'C:\Users\qyu38\Code\RNN_prof_est\model\model_all_route_sim_seq_100_FCL_hatch_only_LSTM_prof_vel_lr0d0001_epoch_200_transfer_mlerp.pth'
# model_path=r'C:\Users\qyu38\Code\RNN_prof_est\model\model_all_route_sim_seq_100_hatch_only_LSTM_prof.pth'
model_path=r'C:\Users\qyu38\Code\RNN_prof_est\model\model_all_route_real_seq_100_hatch_only_prof_min_val_loss_correct.pth'

net.load_state_dict(torch.load(model_path, map_location=device))

net.to(device)  # Move the model to the specified device

batch_size = 4096  # You can adjust the batch size
criterion = nn.MSELoss()  # Mean Squared Error Loss

test_data = TensorDataset(torch.from_numpy(X_test).float(), torch.from_numpy(y_test).float())
test_loader = DataLoader(test_data, batch_size=batch_size)

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
########################################################################################################################################################################################
## Save the values in a text file
import os
import pandas as pd

# Define the path where the CSV files will be saved
save_path = r'C:\Users\qyu38\Code\RNN_prof_est\performance_plot'


# Extract the file name after the last backslash (without extension)
file_name = os.path.basename(model_path).replace('.pth', '.txt')

# Ensure the directory exists
os.makedirs(save_path, exist_ok=True)


# Full paths to save the files
predictions_file = os.path.join(save_path, f'predictions_{file_name}')
ground_truth_file = os.path.join(save_path, f'ground_truth_{file_name}')


# Save the predictions and ground truth as separate txt files
np.savetxt(predictions_file, predictions, fmt='%.6f', delimiter=',')
np.savetxt(ground_truth_file, ground_truth, fmt='%.6f', delimiter=',')

print(f"Predictions saved to: {predictions_file}")
print(f"Ground truth saved to: {ground_truth_file}")

breakpoint()