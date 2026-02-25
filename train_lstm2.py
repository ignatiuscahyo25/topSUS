import numpy as np
import os
import time
import matplotlib.pyplot as plt
import json 

class LSTM_Adam:
    def __init__(self, input_dim, hidden_dim, output_dim, learning_rate=0.001):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.lr = learning_rate
        

        std = 1.0 / np.sqrt(hidden_dim)
        combined_dim = input_dim + hidden_dim
        
        self.Wf = np.random.uniform(-std, std, (hidden_dim, combined_dim))
        self.bf = np.zeros((hidden_dim, 1))
        self.Wi = np.random.uniform(-std, std, (hidden_dim, combined_dim))
        self.bi = np.zeros((hidden_dim, 1))
        self.Wc = np.random.uniform(-std, std, (hidden_dim, combined_dim))
        self.bc = np.zeros((hidden_dim, 1))
        self.Wo = np.random.uniform(-std, std, (hidden_dim, combined_dim))
        self.bo = np.zeros((hidden_dim, 1))
        self.Wy = np.random.uniform(-std, std, (output_dim, hidden_dim))
        self.by = np.zeros((output_dim, 1))
        
        # adam optimizer var
        self.beta1, self.beta2, self.epsilon = 0.9, 0.999, 1e-8
        self.t = 0
        self.params = [self.Wf, self.bf, self.Wi, self.bi, self.Wc, self.bc, self.Wo, self.bo, self.Wy, self.by]
        self.m = [np.zeros_like(p) for p in self.params]
        self.v = [np.zeros_like(p) for p in self.params]

    def sigmoid(self, x): return 1 / (1 + np.exp(-x))
    def tanh(self, x): return np.tanh(x)

    def forward(self, inputs):
        self.inputs = inputs
        T = len(inputs)
        self.h_states = { -1: np.zeros((self.hidden_dim, 1)) }
        self.c_states = { -1: np.zeros((self.hidden_dim, 1)) }
        self.cache = {}
        
        h_t = self.h_states[-1]
        c_t = self.c_states[-1]
        
        for t in range(T):
            x_t = inputs[t].reshape(-1, 1)
            concat = np.vstack((h_t, x_t))
            
            f = self.sigmoid(np.dot(self.Wf, concat) + self.bf)
            i = self.sigmoid(np.dot(self.Wi, concat) + self.bi)
            c_hat = self.tanh(np.dot(self.Wc, concat) + self.bc)
            o = self.sigmoid(np.dot(self.Wo, concat) + self.bo)
            
            c_next = f * c_t + i * c_hat
            h_next = o * self.tanh(c_next)
            
            self.h_states[t] = h_next
            self.c_states[t] = c_next
            self.cache[t] = (f, i, c_hat, o, concat)
            
            h_t, c_t = h_next, c_next
            
        y_pred = np.dot(self.Wy, h_t) + self.by
        return y_pred

    def backward(self, dy):
        T = len(self.inputs)
        dWy, dby = np.dot(dy, self.h_states[T-1].T), dy
        
        grads = [np.zeros_like(p) for p in self.params]
        grads[8], grads[9] = dWy, dby # Wy, by index 8,9
        
        dh_next = np.dot(self.Wy.T, dy)
        dc_next = np.zeros_like(self.c_states[0])
        
        for t in reversed(range(T)):
            h_curr, c_prev = self.h_states[t], self.c_states[t-1]
            f, i, c_hat, o, concat = self.cache[t]
            tc = self.tanh(self.c_states[t])
            
            do = dh_next * tc * o * (1 - o)
            dc = dh_next * o * (1 - tc**2) + dc_next
            di = dc * c_hat * i * (1 - i)
            d_c_hat = dc * i * (1 - c_hat**2)
            df = dc * c_prev * f * (1 - f)
            
            grads[6] += do @ concat.T; grads[7] += do # Wo, bo
            grads[4] += d_c_hat @ concat.T; grads[5] += d_c_hat # Wc, bc
            grads[2] += di @ concat.T; grads[3] += di # Wi, bi
            grads[0] += df @ concat.T; grads[1] += df # Wf, bf
            
            d_concat = (self.Wo.T @ do + self.Wc.T @ d_c_hat + self.Wi.T @ di + self.Wf.T @ df)
            dh_next = d_concat[:self.hidden_dim, :]
            dc_next = dc * f
            
        for g in grads: np.clip(g, -5, 5, out=g)
        self.adam_update(grads)

    def adam_update(self, grads):
        self.t += 1
        for k in range(len(self.params)):
            self.m[k] = self.beta1 * self.m[k] + (1 - self.beta1) * grads[k]
            self.v[k] = self.beta2 * self.v[k] + (1 - self.beta2) * (grads[k]**2)
            m_hat = self.m[k] / (1 - self.beta1**self.t)
            v_hat = self.v[k] / (1 - self.beta2**self.t)
            self.params[k] -= self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)

# Load data n normalisasi
script_dir = os.path.dirname(os.path.abspath(__file__))
train_filename = 'lorenz_train.txt' 
file_path = os.path.join(script_dir, 'topSUS', train_filename) 

print(f"--- 1. Memuat Data: {file_path} ---")

try:
    data_train_raw = np.loadtxt(file_path)
    print(f"Data Loaded. Shape: {data_train_raw.shape}")
except OSError:
    print(f"ERROR: File tidak ditemukan di jalur: {file_path}")
    print("Pastikan nama folder 'topSUS' sudah benar.")
    exit()

min_val = np.min(data_train_raw, axis=0)
max_val = np.max(data_train_raw, axis=0)
denom = max_val - min_val
denom[denom == 0] = 1e-8 

data_train = (data_train_raw - min_val) / denom

# Windowing
TIME_STEP = 5
def create_dataset(dataset, look_back=5):
    X, Y = [], []
    for i in range(len(dataset) - look_back):
        X.append(dataset[i:(i + look_back)])
        Y.append(dataset[i + look_back])
    return np.array(X), np.array(Y)

X_train, y_train = create_dataset(data_train, TIME_STEP)
print(f"Training Samples: {X_train.shape[0]}")

# Training
EPOCHS = 300
HIDDEN_UNITS = 32
model = LSTM_Adam(input_dim=3, hidden_dim=HIDDEN_UNITS, output_dim=3, learning_rate=0.001)

print(f"\n--- 2. Mulai Training ({EPOCHS} Epochs) ---")
start_time = time.time()

loss_history = []
for epoch in range(EPOCHS):
    total_loss = 0
    
    # Shuffle data
    idx = np.arange(len(X_train))
    np.random.shuffle(idx)
    
    for i in idx:
        x_sample = X_train[i]
        y_true = y_train[i].reshape(-1, 1)
        
        # Forward
        y_pred = model.forward(x_sample)
        
        # Loss MSE
        loss = np.mean((y_true - y_pred)**2)
        total_loss += loss
        
        # Backward
        dy = 2 * (y_pred - y_true) / 3
        model.backward(dy)

    avg_loss = total_loss / len(X_train)
    loss_history.append(avg_loss)
    
    if (epoch+1) % 10 == 0 or epoch == 0:
        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {total_loss/len(X_train):.6f}")

print(f"Training Selesai dalam {time.time()-start_time:.2f} detik.")

# Save model
print("\n--- Saving Model to JSON ---")

model_data = {
    "config": {
        "input_dim": 3,
        "hidden_dim": HIDDEN_UNITS,
        "output_dim": 3,
        "time_step": TIME_STEP
    },
    "scaler": {
        "min_val": min_val.tolist(),
        "max_val": max_val.tolist()
    },
    "weights": {
        "Wf": model.Wf.tolist(), "bf": model.bf.tolist(),
        "Wi": model.Wi.tolist(), "bi": model.bi.tolist(),
        "Wc": model.Wc.tolist(), "bc": model.bc.tolist(),
        "Wo": model.Wo.tolist(), "bo": model.bo.tolist(),
        "Wy": model.Wy.tolist(), "by": model.by.tolist()
    }
}

json_path = os.path.join(script_dir, 'model_lorenz.json')
with open(json_path, 'w') as f:
    json.dump(model_data, f, indent=4)

print(f"Model berhasil disimpan ke: {json_path}")

plt.figure(figsize=(10, 5))
plt.plot(range(1, EPOCHS + 1), loss_history, color='blue', linewidth=2)
plt.title('Training Loss (MSE) Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Mean Squared Error')
plt.grid(True)
plt.tight_layout()
plt.show()