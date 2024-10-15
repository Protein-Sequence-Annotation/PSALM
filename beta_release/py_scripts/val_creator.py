import pickle
import random

# Load the original pickle file
with open('datasets/Train_targets_1b_merged.pkl', 'rb') as f:
    data = pickle.load(f)

# Ensure we have a dictionary
if not isinstance(data, dict):
    raise ValueError("The loaded data is not a dictionary!")

# Sample 1000 keys from the dictionary without replacement
sampled_keys = random.sample(list(data.keys()), min(1000, len(data)))

# Create a new dictionary with the sampled key-value pairs
sampled_data = {key: data[key] for key in sampled_keys}

# Save the sampled dictionary into a new pickle file
with open('datasets/Val_targets_1b_final.pkl', 'wb') as f:
    pickle.dump(sampled_data, f)

with open('datasets/val_keys.txt', 'w') as file:
    for key in sampled_keys:
        file.write(f"{key}\n")

print("Sampled dictionary saved to 'datasets/Val_targets_1b_final.pkl'")