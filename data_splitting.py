import numpy as np

# 1. Load the data
# We use loadtxt as the data provided is space-separated floating point numbers
filename = 'lorenz_data.txt'
data = np.loadtxt(filename)

# 2. Shuffle the data
# We generate a list of indices [0, 1, 2... N], shuffle them, 
# and then reorder the data array based on these indices.
indices = np.arange(data.shape[0])
np.random.seed(42) # Set seed for reproducibility
np.random.shuffle(indices)
shuffled_data = data[indices]

# 3. Calculate the split index
split_ratio = 0.8
split_index = int(data.shape[0] * split_ratio)

# 4. Slice the array
train_data = shuffled_data[:split_index]
test_data = shuffled_data[split_index:]

# 5. Save the outputs (Optional)
np.savetxt('train_data.txt', train_data)
np.savetxt('test_data.txt', test_data)

# Print verification
print(f"Original shape: {data.shape}")
print(f"Training shape (80%): {train_data.shape}")
print(f"Testing shape (20%): {test_data.shape}")