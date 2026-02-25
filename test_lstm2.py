import numpy as np
import matplotlib.pyplot as plt
import os
import json

class LSTM_Inference:
    def __init__(self, weights):
        self.Wf = np.array(weights['Wf'])
        self.bf = np.array(weights['bf'])
        self.Wi = np.array(weights['Wi'])
        self.bi = np.array(weights['bi'])
        self.Wc = np.array(weights['Wc'])
        self.bc = np.array(weights['bc'])
        self.Wo = np.array(weights['Wo'])
        self.bo = np.array(weights['bo'])
        self.Wy = np.array(weights['Wy'])
        self.by = np.array(weights['by'])
        
        self.hidden_dim = self.Wf.shape[0]

    def sigmoid(self, x): return 1 / (1 + np.exp(-x))
    def tanh(self, x): return np.tanh(x)

    def predict(self, inputs):
        h_t = np.zeros((self.hidden_dim, 1))
        c_t = np.zeros((self.hidden_dim, 1))
        
        for t in range(len(inputs)):
            x_t = inputs[t].reshape(-1, 1) 
            concat = np.vstack((h_t, x_t))
            
            f = self.sigmoid(np.dot(self.Wf, concat) + self.bf)
            i = self.sigmoid(np.dot(self.Wi, concat) + self.bi)
            c_hat = self.tanh(np.dot(self.Wc, concat) + self.bc)
            o = self.sigmoid(np.dot(self.Wo, concat) + self.bo)
            
            c_next = f * c_t + i * c_hat
            h_next = o * self.tanh(c_next)
            
            h_t, c_t = h_next, c_next
            
        y_pred = np.dot(self.Wy, h_t) + self.by
        return y_pred

# Load model n data
script_dir = os.path.dirname(os.path.abspath(__file__))

json_path = os.path.join(script_dir, 'model_lorenz.json')
if not os.path.exists(json_path):
    print("Error: File 'model_lorenz.json' tidak ditemukan.")
    exit()

with open(json_path, 'r') as f:
    model_data = json.load(f)

print("Model JSON Loaded.")

test_filename = 'lorenz_test.txt' 
test_path = os.path.join(script_dir, 'topSUS', test_filename)

try:
    test_raw = np.loadtxt(test_path)
    print(f"Test Data Loaded from {test_filename}: {test_raw.shape}")
except OSError:
    print(f"Error: File {test_filename} tidak ditemukan di folder topSUS.")
    exit()

# Preprocessing
min_val = np.array(model_data['scaler']['min_val'])
max_val = np.array(model_data['scaler']['max_val'])

denom = max_val - min_val
denom[denom == 0] = 1e-8

test_norm = (test_raw - min_val) / denom

TIME_STEP = model_data['config']['time_step']
X_test, y_test = [], []

for i in range(len(test_norm) - TIME_STEP):
    X_test.append(test_norm[i:(i + TIME_STEP)])
    y_test.append(test_norm[i + TIME_STEP])

X_test = np.array(X_test)
y_test = np.array(y_test)

print(f"Siap untuk testing. Jumlah sampel: {len(X_test)}\n")

# Inference
lstm = LSTM_Inference(model_data['weights'])

predictions = []
print(f"Running inference on {len(X_test)} samples...")

for i in range(len(X_test)):
    pred = lstm.predict(X_test[i])
    predictions.append(pred.flatten())

predictions = np.array(predictions)

# Denormalisasi
pred_denorm = predictions * denom + min_val
actual_denorm = y_test * denom + min_val

# MSE n MAE 
mse_total = np.mean((actual_denorm - pred_denorm)**2)
mae_per_axis = np.mean(np.abs(actual_denorm - pred_denorm), axis=0)
total_mae = np.mean(mae_per_axis)

print("-" * 40)
print(f"Evaluation Results on {test_filename}")
print("-" * 40)
print(f"MAE (X): {mae_per_axis[0]:.6f}")
print(f"MAE (Y): {mae_per_axis[1]:.6f}")
print(f"MAE (Z): {mae_per_axis[2]:.6f}")
print(f"Total MAE: {total_mae:.6f}")
print(f"Total MSE: {mse_total:.6f}")
print("-" * 40)

# Visualisasi
plt.figure(figsize=(10, 5))
plt.plot(actual_denorm[:, 0], label='Actual X', color='black', alpha=0.7)
plt.plot(pred_denorm[:, 0], label='Predicted X', color='red', linestyle='--', linewidth=1.5)
plt.title(f"Lorenz X-Coordinate Prediction\nMAE: {mae_per_axis[0]:.4f}")
plt.xlabel("Time Step")
plt.ylabel("Value X")
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.figure(figsize=(10, 5))
plt.plot(actual_denorm[:, 1], label='Actual Y', color='black', alpha=0.7)
plt.plot(pred_denorm[:, 1], label='Predicted Y', color='green', linestyle='--', linewidth=1.5)
plt.title(f"Lorenz Y-Coordinate Prediction\nMAE: {mae_per_axis[1]:.4f}")
plt.xlabel("Time Step")
plt.ylabel("Value Y")
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.figure(figsize=(10, 5))
plt.plot(actual_denorm[:, 2], label='Actual Z', color='black', alpha=0.7)
plt.plot(pred_denorm[:, 2], label='Predicted Z', color='blue', linestyle='--', linewidth=1.5)
plt.title(f"Lorenz Z-Coordinate Prediction\nMAE: {mae_per_axis[2]:.4f}")
plt.xlabel("Time Step")
plt.ylabel("Value Z")
plt.legend()
plt.grid(True)
plt.tight_layout()

fig4 = plt.figure(figsize=(10, 8))
ax = fig4.add_subplot(111, projection='3d')
ax.plot(actual_denorm[:, 0], actual_denorm[:, 1], actual_denorm[:, 2], 
        lw=0.5, color='black', label='Actual', alpha=0.5)
ax.plot(pred_denorm[:, 0], pred_denorm[:, 1], pred_denorm[:, 2], 
        lw=1, color='red', label='Predicted', linestyle='--')

ax.set_title("Reconstructed Lorentz Attractor")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.legend()
plt.tight_layout()
plt.show()