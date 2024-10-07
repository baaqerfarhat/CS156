


#Q1 to 5 were done on paper

#Q6
unknown_points = [(1, 0, 1), (1, 1, 1), (0, 1, 1)] # 3 Possible points left out

def hypothesis_a(x):
    return 1  

def hypothesis_b(x):
    return 0 

def hypothesis_c(x):
    return (x[0] + x[1] + x[2]) % 2

def hypothesis_d(x):
    return 1 - (x[0] + x[1] + x[2]) % 2

hypotheses = [hypothesis_a, hypothesis_b, hypothesis_c, hypothesis_d]

possible_target_functions = [
    (0, 0, 0),
    (0, 0, 1),
    (0, 1, 0),
    (0, 1, 1),
    (1, 0, 0),
    (1, 0, 1),
    (1, 1, 0),
    (1, 1, 1)
]

def compute_score(hypothesis):
    score = 0
    for target_function in possible_target_functions:
        matches = 0
        for i, point in enumerate(unknown_points):
            if hypothesis(point) == target_function[i]:
                matches += 1
        if matches == 3:
            score += 3
        elif matches == 2:
            score += 2
        elif matches == 1:
            score += 1
    return score

for i, hypothesis in enumerate(hypotheses, 1):
    score = compute_score(hypothesis)
    print(f"Hypothesis {i} has a score of {score}")



#Hypothesis 1 has a score of 12
#Hypothesis 2 has a score of 12
#Hypothesis 3 has a score of 12
#Hypothesis 4 has a score of 12
# Therefore E is the correct answer


#Q7
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

def generated_data(f, N):
    x_random = np.random.uniform(-1, 1, (N, 2))  
    label = np.array([f(x) for x in x_random]) 
    return x_random, label

def PLA(X_random, label):
    N = len(label)  
    weights = np.zeros(3)  
    bias = np.hstack((np.ones((N, 1)), X_random)) 
    its = 0  

    def sign(x): 
        return 1 if np.dot(weights, x) > 0 else -1

    while True:
        misc_points = []
        for i in range(N):
            if sign(bias[i]) != label[i]:  
                misc_points.append(i)

        if not misc_points: 
            break

        random_choice = random.choice(misc_points)
        weights += label[random_choice] * bias[random_choice] 
        its += 1  

    return weights, its  

def estimate_disagreement(f, g, num_test_points=1000):
    X_test = np.random.uniform(-1, 1, (num_test_points, 2)) 
    disagreements = 0
    
    for x in X_test:
        if f(x) != g(x):  
            disagreements += 1
    
    return disagreements / num_test_points  

def hypothesis_g(weights):
    def g(x):
        bias2 = np.hstack(([1], x)) 
        return 1 if np.dot(weights, bias2) > 0 else -1
    return g

def test(N, num_runs=1000):
    total_iterations = 0
    disagreement_prob = 0
    
    for _ in range(num_runs):
        f = target_f()  
        X, y = generated_data(f, N) 
        weights, iterations = PLA(X, y)  
        
        g = hypothesis_g(weights) 
        disagreement_prob += estimate_disagreement(f, g) 
        
        total_iterations += iterations
    
    avg_disagreement = disagreement_prob / num_runs  
    return avg_disagreement

N = 100
disagreement_probability = test(N)
print(f"Estimated disagreement probability: {disagreement_probability}")

"""
N = 100 
avg_iterations = run_experiment(N)
print(f"Average iterations to converge for N = {N}: {avg_iterations}")
"""


#Therefore the answer to 7 is B
#Therefore the answer to 8 is c
#Therefore the answer to 9 is b
#Therefore asnwer to 10 is b 

    

