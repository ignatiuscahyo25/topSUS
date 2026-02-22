import numpy as np
import matplotlib.pyplot as plt

# 1. Memuat data dari file .txt
# Ganti 'data_lorenz.txt' dengan nama file yang Anda miliki
file_path = 'lorenz_data.txt'
data = np.loadtxt(file_path)

# Memisahkan koordinat x, y, dan z
x = data[:, 0]
y = data[:, 1]
z = data[:, 2]

# Membuat array waktu (asumsi dt=0.01 seperti di kode sebelumnya)
dt = 0.01
t = np.arange(len(data)) * dt

# --- VISUALISASI ---

# A. State Trajectory (Time Series) - Menampilkan perilaku x, y, z terhadap waktu
# Sesuai dengan deskripsi pada Figure 1 di paper [cite: 52, 183]
fig1, axs = plt.subplots(3, 1, figsize=(10, 8))
axs[0].plot(t, x, color='blue', lw=1)
axs[0].set_title('x Line Plot')
axs[0].set_ylabel('x(t)')

axs[1].plot(t, y, color='blue', lw=1)
axs[1].set_title('y Line Plot')
axs[1].set_ylabel('y(t)')

axs[2].plot(t, z, color='blue', lw=1)
axs[2].set_title('z Line Plot')
axs[2].set_ylabel('z(t)')
axs[2].set_xlabel('t')
plt.tight_layout()

# B. Two-Dimensional Trajectory - Proyeksi antar sumbu
# Sesuai dengan deskripsi pada Figure 2 di paper [cite: 52, 184]
fig2, axs2 = plt.subplots(1, 3, figsize=(15, 5))
axs2[0].plot(x, y, color='blue', lw=0.7)
axs2[0].set_title('x-y Line Plot')
axs2[0].set_xlabel('x')
axs2[0].set_ylabel('y')

axs2[1].plot(x, z, color='blue', lw=0.7)
axs2[1].set_title('x-z Line Plot')
axs2[1].set_xlabel('x')
axs2[1].set_ylabel('z')

axs2[2].plot(y, z, color='blue', lw=0.7)
axs2[2].set_title('y-z Line Plot')
axs2[2].set_xlabel('y')
axs2[2].set_ylabel('z')
plt.tight_layout()

# C. Three-Dimensional Trajectory - Visualisasi 3D utuh
# Sesuai dengan deskripsi pada Figure 3 di paper [cite: 52, 205]
fig3 = plt.figure(figsize=(10, 8))
ax = fig3.add_subplot(111, projection='3d')
ax.plot(x, y, z, color='blue', lw=0.7)
ax.set_title('x-y-z 3D Trajectory')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')

plt.show()