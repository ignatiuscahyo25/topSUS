import numpy as np

class BPNN_Predictor:
    def __init__(self, weights_path):
        # Load weights and scaler stats
        data = np.load(weights_path)
        self.W1 = data['W1']
        self.b1 = data['b1']
        self.W2 = data['W2']
        self.b2 = data['b2']
        self.mean = data['mean']
        self.std = data['std']
        
    def relu(self, Z):
        return np.maximum(0, Z)

    def predict(self, X_input):
        # Forward Pass
        Z1 = np.dot(X_input, self.W1) + self.b1
        A1 = self.relu(Z1)
        Z2 = np.dot(A1, self.W2) + self.b2
        return Z2 # Returns Scaled Prediction

    def inverse_transform(self, y_scaled):
        # Convert scaled output back to real world XYZ
        return (y_scaled * self.std) + self.mean

def load_test_data(filename, scaler_mean, scaler_std, look_back=5):
    try:
        data = np.loadtxt(filename)
    except OSError:
        print(f"Error: {filename} not found.")
        exit()
        
    # Normalize using TRAINING stats
    data_scaled = (data - scaler_mean) / scaler_std
    
    X, y_real = [], []
    
    for i in range(len(data_scaled) - look_back):
        X.append(data_scaled[i:i+look_back].flatten())
        y_real.append(data[i+look_back]) # Store Ground Truth (Not scaled)
        
    return np.array(X), np.array(y_real)

# --- Main Test Flow ---
if __name__ == "__main__":
    MODEL_FILE = "bpnn_model.npz"
    TEST_FILENAME = "lorenz_train.txt" 
    
    print(f"Loading model from {MODEL_FILE}...")
    try:
        model = BPNN_Predictor(MODEL_FILE)
    except FileNotFoundError:
        print("Model file not found. Please run the training script first.")
        exit()
        
    print(f"Loading and processing test data from {TEST_FILENAME}...")
    X_test, y_true = load_test_data(TEST_FILENAME, model.mean, model.std, look_back=5)
    
    print("Running predictions...")
    # 1. Predict (Result is scaled)
    y_pred_scaled = model.predict(X_test)
    
    # 2. Inverse Scale (Convert back to real coordinates)
    y_pred_final = model.inverse_transform(y_pred_scaled)
    
    # --- PERBAIKAN DI SINI ---
    # Hitung selisih absolut (Absolute Error)
    abs_diff = np.abs(y_true - y_pred_final)
    
    # Hitung MAE per kolom (Axis 0 = rata-rata baris, Axis 1 = kolom)
    # Kolom 0 = X, Kolom 1 = Y, Kolom 2 = Z
    mae_x = np.mean(abs_diff[:, 0])
    mae_y = np.mean(abs_diff[:, 1])
    mae_z = np.mean(abs_diff[:, 2])
    
    # Hitung MAE Total (Rata-rata dari semuanya)
    mae_total = np.mean(abs_diff)
    
    print("-" * 35)
    print(f"Test Results on {len(X_test)} samples:")
    print("-" * 35)
    print(f"MAE (X)     : {mae_x:.6f}")
    print(f"MAE (Y)     : {mae_y:.6f}")
    print(f"MAE (Z)     : {mae_z:.6f}")
    print(f"Total MAE   : {mae_total:.6f}")
    print("-" * 35)
    
    # Show first 5 comparisons
    print("\nSample comparisons (True vs Predicted):")
    print(f"{'True X':<10} {'Pred X':<10} | {'True Y':<10} {'Pred Y':<10} | {'True Z':<10} {'Pred Z':<10}")
    for i in range(5):
        t = y_true[i]
        p = y_pred_final[i]
        print(f"{t[0]:.4f}     {p[0]:.4f}     | {t[1]:.4f}     {p[1]:.4f}     | {t[2]:.4f}     {p[2]:.4f}")