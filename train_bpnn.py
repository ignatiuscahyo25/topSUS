import numpy as np
import time

class BPNN_Adam:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.001):
        self.lr = learning_rate
        
        # He Initialization for ReLU
        self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2. / input_size)
        self.b1 = np.zeros((1, hidden_size))
        
        # Xavier/Glortot Initialization for Output (Linear)
        self.W2 = np.random.randn(hidden_size, output_size) * np.sqrt(1. / hidden_size)
        self.b2 = np.zeros((1, output_size))
        
        # Adam Optimizer parameters
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.epsilon = 1e-8
        self.t = 0 # Timestep
        
        # Initialize moments for Adam
        self.m = {'W1': 0, 'b1': 0, 'W2': 0, 'b2': 0}
        self.v = {'W1': 0, 'b1': 0, 'W2': 0, 'b2': 0}

    def relu(self, Z):
        return np.maximum(0, Z)

    def relu_derivative(self, Z):
        return Z > 0

    def forward(self, X):
        # Layer 1
        self.Z1 = np.dot(X, self.W1) + self.b1
        self.A1 = self.relu(self.Z1)
        
        # Layer 2 (Output)
        self.Z2 = np.dot(self.A1, self.W2) + self.b2
        self.A2 = self.Z2 # Linear activation for regression
        return self.A2

    def backward(self, X, y_true, y_pred):
        m = y_true.shape[0]
        
        # Output Layer Gradients (MSE Derivative)
        # dLoss/dOutput = 2/m * (y_pred - y_true)
        dZ2 = (2 / m) * (y_pred - y_true)
        dW2 = np.dot(self.A1.T, dZ2)
        db2 = np.sum(dZ2, axis=0, keepdims=True)
        
        # Hidden Layer Gradients
        dA1 = np.dot(dZ2, self.W2.T)
        dZ1 = dA1 * self.relu_derivative(self.Z1)
        dW1 = np.dot(X.T, dZ1)
        db1 = np.sum(dZ1, axis=0, keepdims=True)
        
        self.update_parameters(dW1, db1, dW2, db2)
        
        loss = np.mean((y_true - y_pred) ** 2)
        return loss

    def update_parameters(self, dW1, db1, dW2, db2):
        self.t += 1
        grads = {'W1': dW1, 'b1': db1, 'W2': dW2, 'b2': db2}
        params = {'W1': self.W1, 'b1': self.b1, 'W2': self.W2, 'b2': self.b2}
        
        for key in params.keys():
            # Update biased first moment estimate
            self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * grads[key]
            # Update biased second raw moment estimate
            self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * (grads[key] ** 2)
            
            # Compute bias-corrected first moment estimate
            m_hat = self.m[key] / (1 - self.beta1 ** self.t)
            # Compute bias-corrected second raw moment estimate
            v_hat = self.v[key] / (1 - self.beta2 ** self.t)
            
            # Update parameters
            params[key] -= self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)
            
        # Write back to self
        self.W1, self.b1 = params['W1'], params['b1']
        self.W2, self.b2 = params['W2'], params['b2']

def prepare_data(filename, look_back=5):
    # Load data
    try:
        data = np.loadtxt(filename)
    except OSError:
        print(f"Error: {filename} not found. Please ensure the file exists.")
        exit()

    # Normalize data (Standardization) to help Adam converge faster
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    data_scaled = (data - mean) / std

    X, y = [], []
    for i in range(len(data_scaled) - look_back):
        # Flatten the previous 5 samples (5 * 3 = 15 features)
        X.append(data_scaled[i:i+look_back].flatten())
        # Predict the next XYZ
        y.append(data_scaled[i+look_back])
        
    return np.array(X), np.array(y), mean, std

# --- Main Training Flow ---
if __name__ == "__main__":
    FILENAME = "lorenz_train.txt"
    LOOK_BACK = 5
    HIDDEN_NODES = 64
    EPOCHS = 300
    
    print("Loading and processing data...")
    X_train, y_train, train_mean, train_std = prepare_data(FILENAME, LOOK_BACK)
    
    input_dim = X_train.shape[1] # Should be 5 * 3 = 15
    output_dim = y_train.shape[1] # Should be 3
    
    print(f"Input Features: {input_dim}, Output Features: {output_dim}")
    print(f"Training Samples: {len(X_train)}")

    # Initialize Model
    nn = BPNN_Adam(input_dim, HIDDEN_NODES, output_dim, learning_rate=0.001)
    
    start_time = time.time()
    
    print("\nStarting Training...")
    for epoch in range(EPOCHS):
        # Forward pass
        y_pred = nn.forward(X_train)
        
        # Backward pass (Train)
        loss = nn.backward(X_train, y_train, y_pred)
        
        if (epoch + 1) % 50 == 0:
            print(f"Epoch {epoch+1}/{EPOCHS}, Loss (MSE): {loss:.6f}")
            
    print(f"Training Complete. Time: {time.time() - start_time:.2f}s")
    
    # Save model weights and scaler info using np.savez (Robust binary format)
    outfile = "bpnn_model.npz"
    np.savez(outfile, 
             W1=nn.W1, b1=nn.b1, 
             W2=nn.W2, b2=nn.b2, 
             mean=train_mean, std=train_std)
    
    print(f"Model saved to {outfile}")