import numpy as np
import sys

# --- CONFIGURATION ---
INPUT_FILE = "lorenz_train.txt" # Using the same file to test accuracy on training set
MODEL_FILE = "bpnn_model.npz"
WINDOW_SIZE = 5

class BPNN_Predictor:
    def __init__(self, model_path):
        # Load weights and scaler params
        try:
            data = np.load(model_path)
            self.W1 = data['W1']
            self.b1 = data['b1']
            self.W2 = data['W2']
            self.b2 = data['b2']
            self.W3 = data['W3']
            self.b3 = data['b3']
            self.mean = data['mean']
            self.std = data['std']
        except FileNotFoundError:
            print(f"Error: Model file {model_path} not found. Run training first.")
            sys.exit()

    def relu(self, Z):
        return np.maximum(0, Z)

    def predict(self, X):
        # Forward pass only
        Z1 = np.dot(X, self.W1) + self.b1
        A1 = self.relu(Z1)
        
        Z2 = np.dot(A1, self.W2) + self.b2
        A2 = self.relu(Z2)
        
        Z3 = np.dot(A2, self.W3) + self.b3
        # No activation on output (Linear)
        return Z3

    def inverse_transform(self, data):
        return (data * self.std) + self.mean

def load_and_prep_data(filename, mean, std, look_back):
    try:
        data = np.loadtxt(filename)
    except OSError:
        print(f"Error: File {filename} not found.")
        sys.exit()
        
    # Scale using the saved parameters
    data_scaled = (data - mean) / std
    
    X, Y_true = [], []
    for i in range(len(data_scaled) - look_back):
        a = data_scaled[i:(i + look_back)]
        X.append(a.flatten())
        # Keep Y unscaled for accuracy check later? 
        # Better to keep Y scaled for prediction, then inverse both.
        Y_true.append(data[i + look_back]) 
        
    return np.array(X), np.array(Y_true)

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    print(f"Loading model from {MODEL_FILE}...")
    model = BPNN_Predictor(MODEL_FILE)
    
    print(f"Loading and processing test data from {INPUT_FILE}...")
    X_test, Y_true_original = load_and_prep_data(INPUT_FILE, model.mean, model.std, WINDOW_SIZE)
    
    print("Running predictions...")
    # Predict (Output is scaled)
    predictions_scaled = model.predict(X_test)
    
    # Inverse transform predictions to original scale
    predictions_real = model.inverse_transform(predictions_scaled)
    
    # --- EVALUATION ---
    print("\n" + "="*30)
    print("Model Evaluation (MAE)")
    print("="*30)
    
    # Calculate MAE
    # MAE = mean(|y_true - y_pred|)
    abs_errors = np.abs(Y_true_original - predictions_real)
    mae_per_axis = np.mean(abs_errors, axis=0)
    total_mae = np.mean(abs_errors)
    
    print(f"MAE X-axis: {mae_per_axis[0]:.6f}")
    print(f"MAE Y-axis: {mae_per_axis[1]:.6f}")
    print(f"MAE Z-axis: {mae_per_axis[2]:.6f}")
    print("-" * 30)
    print(f"Overall MAE: {total_mae:.6f}")
    print("="*30)
    
    # Show first 5 comparisons
    print("\nSample Comparison (First 5 steps):")
    print(f"{'True (X, Y, Z)':<35} | {'Predicted (X, Y, Z)':<35}")
    print("-" * 73)
    for i in range(5):
        t = Y_true_original[i]
        p = predictions_real[i]
        t_str = f"{t[0]:.2f}, {t[1]:.2f}, {t[2]:.2f}"
        p_str = f"{p[0]:.2f}, {p[1]:.2f}, {p[2]:.2f}"
        print(f"[{t_str:<31}] | [{p_str:<31}]")