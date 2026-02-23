import numpy as np
import json
import os

# --- 1. Data Loading and Preprocessing ---
def load_and_process_data(filename, window_size=5):
    # Load data from text file
    data = np.loadtxt(filename)
    
    # Normalize data (Standard Scaling: (x - mean) / std)
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    data_normalized = (data - mean) / std
    
    X, y = [], []
    # Create windows: 5 steps of (x,y,z) to predict the 6th step
    for i in range(len(data_normalized) - window_size):
        X.append(data_normalized[i : i + window_size])
        y.append(data_normalized[i + window_size])
        
    return np.array(X), np.array(y), mean, std

# --- 2. CNN Layers Implementation ---

def conv1d_forward(X, W, b, stride=1):
    """
    X: Input (Batch, TimeSteps, Channels) -> (N, 5, 3)
    W: Weights (Filters, KernelSize, Channels) -> (8, 3, 3)
    b: Bias (Filters)
    """
    n_samples, n_time, n_channels = X.shape
    n_filters, kernel_size, _ = W.shape
    
    output_len = (n_time - kernel_size) // stride + 1
    output = np.zeros((n_samples, output_len, n_filters))
    
    # Simple unoptimized sliding window loop
    for i in range(output_len):
        start = i * stride
        end = start + kernel_size
        # Slice X: (N, K, C)
        x_slice = X[:, start:end, :]
        
        for f in range(n_filters):
            # Convolve: Sum over Kernel and Channels dimensions
            # x_slice * W[f]: (N, K, C) * (K, C) -> (N, K, C)
            output[:, i, f] = np.sum(x_slice * W[f], axis=(1, 2)) + b[f]
            
    return output

def relu(z):
    return np.maximum(0, z)

def dense_forward(X_flat, W, b):
    # X_flat: (N, InputDim)
    # W: (InputDim, OutputDim)
    return np.dot(X_flat, W) + b

# --- 3. Training Logic ---

def train_model():
    # Hyperparameters
    FILENAME = 'lorenz_train.txt'
    LR = 0.001
    EPOCHS = 300
    WINDOW_SIZE = 5
    BATCH_SIZE = 32
    
    # Model Architecture Params
    INPUT_CHANNELS = 3
    NUM_FILTERS = 16
    KERNEL_SIZE = 3
    
    # Load Data
    print(f"Loading {FILENAME}...")
    if not os.path.exists(FILENAME):
        print(f"Error: {FILENAME} not found. Please create the file with the dataset.")
        return

    X, y, mean, std = load_and_process_data(FILENAME, WINDOW_SIZE)
    
    # Shapes
    # X: (N, 5, 3)
    # Conv Output Length: 5 - 3 + 1 = 3
    # Conv Output Volume: 3 * 16 (Filters) = 48 flattened features
    flatten_dim = 3 * NUM_FILTERS 
    output_dim = 3 # Predicting X, Y, Z
    
    # Initialization (He Initialization for ReLU)
    np.random.seed(42)
    weights = {
        'W_conv': np.random.randn(NUM_FILTERS, KERNEL_SIZE, INPUT_CHANNELS) * np.sqrt(2/(KERNEL_SIZE*INPUT_CHANNELS)),
        'b_conv': np.zeros(NUM_FILTERS),
        'W_dense': np.random.randn(flatten_dim, output_dim) * np.sqrt(2/flatten_dim),
        'b_dense': np.zeros((1, output_dim))
    }
    
    # Adam Optimizer Parameters
    beta1, beta2, epsilon = 0.9, 0.999, 1e-8
    m = {k: np.zeros_like(v) for k, v in weights.items()}
    v = {k: np.zeros_like(v) for k, v in weights.items()}
    t = 0 # Time step
    
    n_samples = X.shape[0]
    
    print("Starting Training...")
    
    for epoch in range(EPOCHS):
        # Shuffle
        indices = np.arange(n_samples)
        np.random.shuffle(indices)
        X_shuffled = X[indices]
        y_shuffled = y[indices]
        
        epoch_loss = 0
        
        for i in range(0, n_samples, BATCH_SIZE):
            X_batch = X_shuffled[i : i + BATCH_SIZE]
            y_batch = y_shuffled[i : i + BATCH_SIZE]
            curr_batch_size = X_batch.shape[0]
            
            # --- Forward Pass ---
            # 1. Conv1D
            Z_conv = conv1d_forward(X_batch, weights['W_conv'], weights['b_conv'])
            # 2. ReLU
            A_conv = relu(Z_conv)
            # 3. Flatten
            A_flat = A_conv.reshape(curr_batch_size, -1)
            # 4. Dense
            Z_out = dense_forward(A_flat, weights['W_dense'], weights['b_dense'])
            
            # --- Loss (MSE) ---
            loss = np.mean((Z_out - y_batch) ** 2)
            epoch_loss += loss * curr_batch_size
            
            # --- Backward Pass ---
            # dLoss/dZ_out (Derivative of MSE)
            dZ_out = 2 * (Z_out - y_batch) / curr_batch_size
            
            # Gradients for Dense Layer
            dW_dense = np.dot(A_flat.T, dZ_out)
            db_dense = np.sum(dZ_out, axis=0, keepdims=True)
            
            # Propagate back to Flatten/Conv
            dA_flat = np.dot(dZ_out, weights['W_dense'].T)
            dA_conv = dA_flat.reshape(curr_batch_size, 3, NUM_FILTERS) # Reshape back to (N, T_out, F)
            
            # Derivative of ReLU
            dZ_conv = dA_conv * (Z_conv > 0)
            
            # Gradients for Conv Layer
            dW_conv = np.zeros_like(weights['W_conv'])
            db_conv = np.zeros_like(weights['b_conv'])
            
            # Compute dW_conv and db_conv (Cross-correlation for gradients)
            # Z_conv shape was (N, 3, F) where 3 is time output
            output_len = Z_conv.shape[1]
            
            for l in range(output_len):
                # Input slice that generated this output step
                x_slice = X_batch[:, l : l + KERNEL_SIZE, :] # (N, K, C)
                
                for f in range(NUM_FILTERS):
                    # dZ_conv[:, l, f] shape is (N,)
                    # We need to sum over batch
                    grad_slice = dZ_conv[:, l, f].reshape(-1, 1, 1) # (N, 1, 1)
                    
                    # Accumulate gradient for the filter
                    dW_conv[f] += np.sum(x_slice * grad_slice, axis=0)
                    db_conv[f] += np.sum(dZ_conv[:, l, f])

            # --- Adam Optimizer Update ---
            t += 1
            grads = {'W_conv': dW_conv, 'b_conv': db_conv, 'W_dense': dW_dense, 'b_dense': db_dense}
            
            for key in weights:
                # Update moving averages
                m[key] = beta1 * m[key] + (1 - beta1) * grads[key]
                v[key] = beta2 * v[key] + (1 - beta2) * (grads[key] ** 2)
                
                # Bias correction
                m_hat = m[key] / (1 - beta1 ** t)
                v_hat = v[key] / (1 - beta2 ** t)
                
                # Update weights
                weights[key] -= LR * m_hat / (np.sqrt(v_hat) + epsilon)
        
        if (epoch + 1) % 50 == 0:
            print(f"Epoch {epoch+1}/{EPOCHS}, Loss (MSE): {epoch_loss/n_samples:.6f}")

    # --- Save Model ---
    model_data = {
        'weights': {k: v.tolist() for k, v in weights.items()},
        'norm': {'mean': mean.tolist(), 'std': std.tolist()},
        'architecture': {'filters': NUM_FILTERS, 'kernel': KERNEL_SIZE}
    }
    
    with open('cnn_model_params.json', 'w') as f:
        json.dump(model_data, f)
    
    print("Training Complete. Model saved to 'cnn_model_params.json'")

if __name__ == "__main__":
    train_model()