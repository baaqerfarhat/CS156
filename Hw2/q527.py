import numpy as np
import random

def target_f():
    dataPoints = np.random.uniform(-1, 1, (2, 2))  
    Point1, Point2 = dataPoints[0], dataPoints[1]
    slope = (Point2[1] - Point1[1]) / (Point2[0] - Point1[0]) 
    yIntercept = Point1[1] - slope * Point1[0]
    
    def f(x): 
        if x[1] > slope * x[0] + yIntercept:
            return 1
        else:
            return -1
    
    return f

def generate_data(f, N):
    x_random = np.random.uniform(-1, 1, (N, 2)) 
    label = np.array([f(x) for x in x_random])  
    return x_random, label

def lin_reg(X, y):
    N = len(y)
    X_bias = np.hstack((np.ones((N, 1)), X))  
    inverse = np.linalg.pinv(X_bias)
    weights = np.dot(inverse, y)
    return weights

def estimate_E_in(weights, X, y):
    X_bias = np.hstack((np.ones((X.shape[0], 1)), X))
    predictions = np.sign(np.dot(X_bias, weights)) 
    misclassified = np.sum(predictions != y) 
    return misclassified / len(y)  

def estimate_E_out(f, weights, N=1000):
    X_fresh, y_fresh = generate_data(f, N) 
    X_bias = np.hstack((np.ones((X_fresh.shape[0], 1)), X_fresh))
    predictions = np.sign(np.dot(X_bias, weights))  
    misclassified = np.sum(predictions != y_fresh)
    return misclassified / len(y_fresh)

def perceptron(X, y, weights):
    X_bias = np.hstack((np.ones((X.shape[0], 1)), X))  
    num_iter = 0
    while True:
        predictions = np.sign(np.dot(X_bias, weights))
        misclassified_indices = np.where(predictions != y)[0]  
        if len(misclassified_indices) == 0:  
            break
        random_index = random.choice(misclassified_indices)
        weights += y[random_index] * X_bias[random_index]  
        num_iter += 1
    return num_iter  

def test(N, num_runs=1000):
    total_E_in = 0
    total_E_out = 0
    total_PLA_iterations = 0

    for _ in range(num_runs):
        f = target_f()  
        X, y = generate_data(f, N)  
        weights = lin_reg(X, y)  

        E_in = estimate_E_in(weights, X, y)
        total_E_in += E_in

        E_out = estimate_E_out(f, weights)
        total_E_out += E_out

        pla_iters = perceptron(X, y, weights)  
        total_PLA_iterations += pla_iters

    avg_E_in = total_E_in / num_runs
    avg_E_out = total_E_out / num_runs
    avg_PLA_iterations = total_PLA_iterations / num_runs

    return avg_E_in, avg_E_out, avg_PLA_iterations

N = 10  
num_runs = 1000  

avg_E_in, avg_E_out, avg_PLA_iterations = test(N, num_runs)

print(f"Average E_in: {avg_E_in}")
print(f"Average E_out: {avg_E_out}")
print(f"Average PLA iterations: {avg_PLA_iterations}")

#5: Average E_in: 0.037899999999999996
#6: Average E_out: 0.047750000000000015
#7: Average PLA iterations: 3.938