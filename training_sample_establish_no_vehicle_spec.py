import torch
import torch.nn as nn
import torch.optim as optim
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

directory_path = r'C:\Users\qyu38\Code\RNN_prof_est\data_all_five_one_vehicle_strict\train'

num_features = 3  # Number of feature columns
sequence_length = 100  # Length of the sequence for each input
processed_features = []
processed_labels = []

# Iterate over the files in the directory
for filename in os.listdir(directory_path):
    if filename.endswith(".txt"):  # Assuming the files are .txt files
        file_path = os.path.join(directory_path, filename)

        # Load the file
        df = pd.read_csv(file_path, sep='\t')  # or whatever your separator is
        # breakpoint()
        # Process the file
        for i in range(len(df) - sequence_length + 1):
            feature_sequence = df.iloc[i:i + sequence_length, -5:-2]
                        # Check if any speed in the sequence is less than 5
            # if not (feature_sequence['speed'] < 5).any(): #remove the low speed region
            #     label = df.iloc[i, -1]
            #     # breakpoint()
            #     #we might need to add a condition here to wipe out features sequences that have speed being less than 5m/s 
            #     processed_features.append(feature_sequence.values)
            #     processed_labels.append(label)
            label = df.iloc[i, -1]    
            processed_features.append(feature_sequence.values)
            processed_labels.append(label)
        print(f"Processed file: {filename}")


# Convert lists to numpy arrays Assuming X is your input matrix with shape [num_samples, 200, 7]
# and y is the array of profile points with shape [num_samples, 1]
X = np.array(processed_features)
y = np.array(processed_labels).reshape(-1, 1)

# Assuming X is your feature matrix of shape [num_samples, 200, 7]
num_samples, sequence_length, num_features = X.shape
# Reshape X to 2D
X_reshaped = X.reshape(-1, num_features)  # New shape will be [num_samples * 200, 7]
# Apply StandardScaler
scaler_X = StandardScaler()
X_standardized_reshaped = scaler_X.fit_transform(X_reshaped)
# Reshape X back to 3D
X_standardized = X_standardized_reshaped.reshape(num_samples, sequence_length, num_features)

# breakpoint()
# Specify the actual folder and filenames
folder_path = 'C:/Users/qyu38/Code/RNN_prof_est/npy_all_five' # Replace with your actual folder path
X_filename = '/X_all_route_aligned_2000_seq_100_train_hatch_only_prof.npy'
y_filename = '/y_all_route_aligned_2000_seq_100_train_hatch_only_prof.npy'

# Saving the arrays
# Attempt to save the file and handle exceptions
try:
    np.save(folder_path + X_filename, X_standardized)
    np.save(folder_path + y_filename, y)
except FileNotFoundError as e:
    print("Error: File not found.", e)
except Exception as e:
    print("An unexpected error occurred:", e)


breakpoint()


