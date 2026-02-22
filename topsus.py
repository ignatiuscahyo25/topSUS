import numpy as np

sigma = 10
beta = 8/3
rho = 28

def lorenz_system(x,y,z):
    px = sigma * (y - x)
    py = x * (rho - z) - y
    pz = x * y - beta * z
    return np.array([px, py, pz])

def runge_kutta_step(func, state, dt):
    k1 = func(*state)
    k2 = func(*(state + 0.5 * dt * k1))
    k3 = func(*(state + 0.5 * dt * k2))
    k4 = func(*(state + dt * k3))
    return state + (dt / 6) * (k1 + 2*k2 + 2*k3 + k4)

dt = 0.01
num_steps = 10000

data = np.zeros((num_steps+1, 3))
data[0] = np.array([1.0, 1.0, 1.0])  # Initial condition

for i in range(num_steps):
    data[i+1]   = runge_kutta_step(lorenz_system, data[i], dt)

np.savetxt('lorenz_data.txt', data)