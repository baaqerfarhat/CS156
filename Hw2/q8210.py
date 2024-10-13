import numpy as np

def generate_points(N):
    X = np.random.uniform(-1, 1, (N, 2))  
    y = np.sign(X[:, 0]**2 + X[:, 1]**2 - 0.6)  
    return X, y

def add_noise(y, noise_percentage=0.1):
    num_noisy_points = int(len(y) * noise_percentage)
    noisy_indices = np.random.choice(len(y), num_noisy_points, replace=False)
    y[noisy_indices] *= -1  
    return y

def linear_regression(X, y):
    X_bias = np.hstack((np.ones((X.shape[0], 1)), X))
    weights = np.linalg.pinv(X_bias) @ y
    return weights

def estimate_E_in(weights, X, y):
    X_bias = np.hstack((np.ones((X.shape[0], 1)), X))
    predictions = np.sign(X_bias @ weights)
    E_in = np.mean(predictions != y)
    return E_in

def transform_features(X):
    x1 = X[:, 0]
    x2 = X[:, 1]
    return np.column_stack([np.ones(X.shape[0]), x1, x2, x1 * x2, x1**2, x2**2])

def linear_regression_transformed(X, y):
    X_transformed = transform_features(X)
    weights = np.linalg.pinv(X_transformed) @ y
    return weights

def estimate_E_out_transformed(weights, N=1000):
    X_fresh, y_fresh = generate_points(N)
    y_fresh = add_noise(y_fresh)
    X_transformed = transform_features(X_fresh)
    predictions = np.sign(X_transformed @ weights)
    E_out = np.mean(predictions != y_fresh)
    return E_out

def problem_8(num_runs=1000, N=1000):
    total_E_in = 0
    for _ in range(num_runs):
        X, y = generate_points(N)
        y = add_noise(y)
        weights = linear_regression(X, y)
        E_in = estimate_E_in(weights, X, y)
        total_E_in += E_in
    return total_E_in / num_runs

def problem_9(num_runs=1000, N=1000):
    total_weights = np.zeros(6)
    for _ in range(num_runs):
        X, y = generate_points(N)
        y = add_noise(y)
        weights = linear_regression_transformed(X, y)
        total_weights += weights
    return total_weights / num_runs

def problem_10(num_runs=1000, N=1000):
    avg_weights = problem_9(num_runs, N)
    total_E_out = 0
    for _ in range(num_runs):
        E_out = estimate_E_out_transformed(avg_weights, N)
        total_E_out += E_out
    return total_E_out / num_runs

avg_E_in_8 = problem_8()
avg_weights_9 = problem_9()
avg_E_out_10 = problem_10()

print(avg_E_in_8)
print(avg_weights_9)
print(avg_E_out_10)

"""
Code output: 
0.5040570000000006
[-9.92287770e-01  1.90048548e-03  1.22809046e-04  2.35387695e-03
  1.55713326e+00  1.55763666e+00]
0.12396900000000008
"""