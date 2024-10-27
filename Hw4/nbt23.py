import numpy as np
import matplotlib.pyplot as plt

d_vc = 50
delta = 0.05
N_values = np.linspace(1, 10000, 1000)

def growth_function(N):
    return N ** d_vc

def vc_bound(N):
    return np.sqrt((8 / N) * np.log(4 * growth_function(2 * N) / delta))

def rademacher_bound(N):
    return np.sqrt((2 * np.log(2 * N * growth_function(N)) / N) + np.sqrt(2 / N * np.log(1 / delta)) + 1 / N)

def parrondo_bound(N):
    return np.sqrt((1 / N) * (2 + np.log(6 * growth_function(2 * N) / delta)))

def devroye_bound(N):
    epsilon = 0.05  
    return np.sqrt((1 / (2 * N)) * (4 * epsilon * (1 + epsilon) + np.log(4 * growth_function(N**2) / delta)))

vc_bounds = vc_bound(N_values)
rademacher_bounds = rademacher_bound(N_values)
parrondo_bounds = parrondo_bound(N_values)
devroye_bounds = devroye_bound(N_values)

plt.figure(figsize=(10, 6))
plt.plot(N_values, vc_bounds, label="Original VC Bound (a)")
plt.plot(N_values, rademacher_bounds, label="Rademacher Penalty Bound (b)")
plt.plot(N_values, parrondo_bounds, label="Parrondo and Van den Broek Bound (c)")
plt.plot(N_values, devroye_bounds, label="Devroye Bound (d)")

plt.xlabel("Sample Size N")
plt.ylabel("Generalization Error ε")
plt.title("Generalization Error Bounds as a Function of Sample Size N")
plt.legend()
plt.grid(True)
plt.show()
