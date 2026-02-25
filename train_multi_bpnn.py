import numpy as np
import sys

# --- CONFIGURATION ---
INPUT_FILE = "lorenz_train.txt"
MODEL_SAVE_FILE = "bpnn_model.npz"
WINDOW_SIZE = 5  # Look back 5 samples
LEARNING_RATE = 0.001
EPOCHS = 300
HIDDEN_1_SIZE = 64
HIDDEN_2_SIZE = 32
OUTPUT_SIZE = 3

# --- HELPER FUNCTIONS ---
def load_data(filename):
    try:
        # Load data ignoring comments or headers if any
        data = np.loadtxt(filename)
        return data
    except OSError:
        print(f"Error: File {filename} not found.")
        sys.exit()

def create_dataset(dataset, look_back=1):
    X, Y = [], []
    for i in range(len(dataset) - look_back):
        # Flatten the 5x3 matrix into a 1D array of size 15
        a = dataset[i:(i + look_back)]
        X.append(a.flatten()) 
        Y.append(dataset[i + look_back])
    return np.array(X), np.array(Y)

def relu(Z):
    return np.maximum(0, Z)

def relu_derivative(Z):
    return Z > 0

# --- BPNN CLASS WITH ADAM ---
class BPNN_MultiLayer:
    def __init__(self, input_dim, h1_dim, h2_dim, output_dim):
        # He Initialization for ReLU
        self.W1 = np.random.randn(input_dim, h1_dim) * np.sqrt(2. / input_dim)
        self.b1 = np.zeros((1, h1_dim))
        
        self.W2 = np.random.randn(h1_dim, h2_dim) * np.sqrt(2. / h1_dim)
        self.b2 = np.zeros((1, h2_dim))
        
        self.W3 = np.random.randn(h2_dim, output_dim) * np.sqrt(2. / h2_dim)
        self.b3 = np.zeros((1, output_dim))

        # Adam Optimizer Parameters
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.epsilon = 1e-8
        self.t = 0 # Time step
        
        # Initialize moments for Adam
        self.m = {k: np.zeros_like(v) for k, v in 
                  {'W1': self.W1, 'b1': self.b1, 'W2': self.W2, 'b2': self.b2, 'W3': self.W3, 'b3': self.b3}.items()}
        self.v = {k: np.zeros_like(v) for k, v in 
                  {'W1': self.W1, 'b1': self.b1, 'W2': self.W2, 'b2': self.b2, 'W3': self.W3, 'b3': self.b3}.items()}

    def forward(self, X):
        # Layer 1
        self.Z1 = np.dot(X, self.W1) + self.b1
        self.A1 = relu(self.Z1)
        
        # Layer 2
        self.Z2 = np.dot(self.A1, self.W2) + self.b2
        self.A2 = relu(self.Z2)
        
        # Output Layer (Linear activation for regression)
        self.Z3 = np.dot(self.A2, self.W3) + self.b3
        self.A3 = self.Z3 # Linear
        return self.A3

    def backward(self, X, y_true, y_pred, lr):
        m = X.shape[0]
        
        # Output Layer Gradients (MSE Derivative w.r.t Linear Output)
        # dL/dZ3 = 2/m * (y_pred - y_true)
        dZ3 = (2/m) * (y_pred - y_true) 
        dW3 = np.dot(self.A2.T, dZ3)
        db3 = np.sum(dZ3, axis=0, keepdims=True)
        
        # Layer 2 Gradients
        dA2 = np.dot(dZ3, self.W3.T)
        dZ2 = dA2 * relu_derivative(self.Z2)
        dW2 = np.dot(self.A1.T, dZ2)
        db2 = np.sum(dZ2, axis=0, keepdims=True)
        
        # Layer 1 Gradients
        dA1 = np.dot(dZ2, self.W2.T)
        dZ1 = dA1 * relu_derivative(self.Z1)
        dW1 = np.dot(X.T, dZ1)
        db1 = np.sum(dZ1, axis=0, keepdims=True)

        # Update Weights using Adam
        self.adam_update({'W1': dW1, 'b1': db1, 'W2': dW2, 'b2': db2, 'W3': dW3, 'b3': db3}, lr)
        
        loss = np.mean((y_true - y_pred) ** 2)
        return loss

    def adam_update(self, gradients, lr):
        self.t += 1
        params = {'W1': self.W1, 'b1': self.b1, 'W2': self.W2, 'b2': self.b2, 'W3': self.W3, 'b3': self.b3}
        
        for key in params.keys():
            # Update first moment estimate
            self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * gradients[key]
            # Update second raw moment estimate
            self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * (gradients[key] ** 2)
            
            # Bias correction
            m_hat = self.m[key] / (1 - self.beta1 ** self.t)
            v_hat = self.v[key] / (1 - self.beta2 ** self.t)
            
            # Update parameters
            params[key] -= lr * m_hat / (np.sqrt(v_hat) + self.epsilon)

        # Reflect changes back to class attributes
        self.W1, self.b1 = params['W1'], params['b1']
        self.W2, self.b2 = params['W2'], params['b2']
        self.W3, self.b3 = params['W3'], params['b3']

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    print("1. Loading Data...")
    raw_data = load_data(INPUT_FILE)
    
    # Normalize Data (Standard Scaling)
    mean = np.mean(raw_data, axis=0)
    std = np.std(raw_data, axis=0)
    data_scaled = (raw_data - mean) / std

    print("2. Creating Windowed Dataset...")
    X, y = create_dataset(data_scaled, WINDOW_SIZE)
    input_dim = X.shape[1] # Should be 5 * 3 = 15

    print(f"   Input shape: {X.shape}, Output shape: {y.shape}")

    # Initialize Model
    nn = BPNN_MultiLayer(input_dim, HIDDEN_1_SIZE, HIDDEN_2_SIZE, OUTPUT_SIZE)
    
    print(f"3. Training for {EPOCHS} epochs with Adam...")
    for epoch in range(EPOCHS):
        # Forward pass
        y_pred = nn.forward(X)
        
        # Backward pass & Update
        loss = nn.backward(X, y, y_pred, LEARNING_RATE)
        
        if (epoch + 1) % 10 == 0:
            print(f"   Epoch {epoch+1}/{EPOCHS} - MSE Loss: {loss:.6f}")

    # Save Model
    print(f"4. Saving model to {MODEL_SAVE_FILE}...")
    np.savez(MODEL_SAVE_FILE, 
             W1=nn.W1, b1=nn.b1, 
             W2=nn.W2, b2=nn.b2, 
             W3=nn.W3, b3=nn.b3,
             mean=mean, std=std)
    print("   Done.")