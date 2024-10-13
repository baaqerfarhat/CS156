import numpy as np

def generate_distributions(numm_coins=1000, numm_flips=10, numm_experiments=100000):
    num_c1_values = []
    num_crand_values = []
    num_cmin_values = []
    
    for _ in range(numm_experiments):
        flips = np.random.randint(0, 2, (numm_coins, numm_flips))
        frequencies = np.mean(flips, axis=1)
        num_c1 = frequencies[0]
        num_c1_values.append(num_c1)
        num_crand = frequencies[np.random.randint(0, numm_coins)]
        num_crand_values.append(num_crand)
        
        num_cmin = np.min(frequencies)
        num_cmin_values.append(num_cmin)
    
    return np.array(num_c1_values), np.array(num_crand_values), np.array(num_cmin_values)

num_c1_values, num_crand_values, num_cmin_values = generate_distributions()

num_c1_mean = np.mean(num_c1_values)
num_crand_mean = np.mean(num_crand_values)
num_cmin_mean = np.mean(num_cmin_values)

print(num_c1_mean, num_crand_mean, num_cmin_mean)

"""
Code Output:
0.500073 0.5002820000000001 0.037601
"""
