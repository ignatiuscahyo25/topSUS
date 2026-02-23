import numpy as np
import json
import os
import matplotlib.pyplot as plt

# --- Helper Functions (Reused logic) ---

def conv1d_forward(X, W, b, stride=1):
    n_samples, n_time, n_channels = X.shape
    n_filters, kernel_size, _ = W.shape
    output_len = (n_time - kernel_size) // stride + 1
    output = np.zeros((n_samples, output_len, n_filters))
    
    for i in range(output_len):
        start = i * stride
        end = start + kernel_size
        x_slice = X[:, start:end, :]
        for f in range(n_filters):
            output[:, i, f] = np.sum(x_slice * W[f], axis=(1, 2)) + b[f]
    return output

def relu(z):
    return np.maximum(0, z)

def dense_forward(X_flat, W, b):
    return np.dot(X_flat, W) + b

# --- Testing Logic ---

def test_model():
    MODEL_FILE = 'cnn_model_params.json'
    
    # If lorenz_test.txt exists, we use it. Otherwise, we test on lorenz_train.txt
    if os.path.exists('lorenz_test.txt'):
        TEST_FILE = 'lorenz_test.txt'
    else:
        TEST_FILE = 'lorenz_train.txt'
        print(f"Note: 'lorenz_test.txt' not found. Testing on '{TEST_FILE}' instead.")

    if not os.path.exists(MODEL_FILE):
        print("Error: Model file not found. Run train_cnn.py first.")
        return

    # 1. Load Model Parameters
    with open(MODEL_FILE, 'r') as f:
        model_data = json.load(f)
    
    # Convert lists back to numpy arrays
    weights = {k: np.array(v) for k, v in model_data['weights'].items()}
    norm_mean = np.array(model_data['norm']['mean'])
    norm_std = np.array(model_data['norm']['std'])
    
    # 2. Load and Preprocess Test Data
    raw_data = np.loadtxt(TEST_FILE)
    
    # Normalize test data using training stats
    data_normalized = (raw_data - norm_mean) / norm_std
    
    WINDOW_SIZE = 5
    X_test, y_test_norm = [], []
    
    # Create windows
    for i in range(len(data_normalized) - WINDOW_SIZE):
        X_test.append(data_normalized[i : i + WINDOW_SIZE])
        y_test_norm.append(data_normalized[i + WINDOW_SIZE])
        
    X_test = np.array(X_test)
    y_test_norm = np.array(y_test_norm)
    y_test_actual = raw_data[WINDOW_SIZE:] # The actual original values
    
    print(f"Running inference on {len(X_test)} samples...")
    
    # 3. Inference (Forward Pass)
    Z_conv = conv1d_forward(X_test, weights['W_conv'], weights['b_conv'])
    A_conv = relu(Z_conv)
    A_flat = A_conv.reshape(X_test.shape[0], -1)
    Z_out = dense_forward(A_flat, weights['W_dense'], weights['b_dense'])
    
    # 4. Denormalize Predictions
    predictions_norm = Z_out
    predictions_actual = (predictions_norm * norm_std) + norm_mean
    
    # 5. Evaluation (MAE)
    mae_per_axis = np.mean(np.abs(y_test_actual - predictions_actual), axis=0)
    total_mae = np.mean(np.abs(y_test_actual - predictions_actual))
    
    print("-" * 30)
    print(f"Evaluation Results on {TEST_FILE}")
    print("-" * 30)
    print(f"MAE (X): {mae_per_axis[0]:.6f}")
    print(f"MAE (Y): {mae_per_axis[1]:.6f}")
    print(f"MAE (Z): {mae_per_axis[2]:.6f}")
    print(f"Total MAE: {total_mae:.6f}")

    # --- 6. Visualization ---
    print("\nGenerating Graphs...")
    
    # Create a time axis for plotting
    time_steps = np.arange(len(y_test_actual))

    # Graph 1: Component-wise comparison over time
    plt.figure(figsize=(15, 10))

    # X Axis Plot
    plt.subplot(3, 1, 1)
    plt.plot(time_steps, y_test_actual[:, 0], label='Actual X', color='blue', linewidth=1)
    plt.plot(time_steps, predictions_actual[:, 0], label='Predicted X', color='red', linestyle='--', linewidth=1)
    plt.title('X Coordinate: Actual vs Predicted')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Y Axis Plot
    plt.subplot(3, 1, 2)
    plt.plot(time_steps, y_test_actual[:, 1], label='Actual Y', color='green', linewidth=1)
    plt.plot(time_steps, predictions_actual[:, 1], label='Predicted Y', color='orange', linestyle='--', linewidth=1)
    plt.title('Y Coordinate: Actual vs Predicted')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Z Axis Plot
    plt.subplot(3, 1, 3)
    plt.plot(time_steps, y_test_actual[:, 2], label='Actual Z', color='purple', linewidth=1)
    plt.plot(time_steps, predictions_actual[:, 2], label='Predicted Z', color='magenta', linestyle='--', linewidth=1)
    plt.title('Z Coordinate: Actual vs Predicted')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # Graph 2: 3D Trajectory comparison
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot Actual
    ax.plot(y_test_actual[:, 0], y_test_actual[:, 1], y_test_actual[:, 2], 
            label='Actual Trajectory', color='blue', alpha=0.6, linewidth=0.8)
            
    # Plot Predicted
    ax.plot(predictions_actual[:, 0], predictions_actual[:, 1], predictions_actual[:, 2], 
            label='Predicted Trajectory', color='red', alpha=0.6, linewidth=0.8)

    ax.set_xlabel("X Axis")
    ax.set_ylabel("Y Axis")
    ax.set_zlabel("Z Axis")
    ax.set_title("3D Lorenz Attractor: Actual vs Predicted")
    ax.legend()
    plt.show()

if __name__ == "__main__":
    test_model()