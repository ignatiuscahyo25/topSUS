import numpy as np

# 1. Load the data
filename = 'lorenz_data.txt'
data = np.loadtxt(filename)

# 2. Calculate the split index
# We take the total number of rows (data.shape[0]) and multiply by 0.8
split_ratio = 0.80
split_index = int(data.shape[0] * split_ratio)

# 3. Split the data sequentially
# train_data takes everything from index 0 up to split_index
# test_data takes everything from split_index to the end
train_data = data[:split_index]
test_data = data[split_index:]

# 4. Save the split files (Optional)
np.savetxt('lorenz_train.txt', train_data)
np.savetxt('lorenz_test.txt', test_data)

# 5. Verify the split
print(f"Total samples: {data.shape[0]}")
print(f"Split index:   {split_index}")
print("-" * 30)
print(f"Training set shape: {train_data.shape} (First 80%)")
print(f"Testing set shape:  {test_data.shape} (Last 20%)")

# Verify continuity (The last point of train should come just before first point of test)
print("-" * 30)
print(f"Last training point:  {train_data[-1]}")
print(f"First testing point:  {test_data[0]}")