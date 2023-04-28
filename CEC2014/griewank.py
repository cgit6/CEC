import math

def griewank(x):
    sum = 0
    product = 1
    for i in range(len(x)):
        sum += x[i]**2
        product *= math.cos(x[i]/math.sqrt(i+1))
    return 1 + (sum/4000) - product