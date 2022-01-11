import numpy as np
import math


def rosenbrock(x):
    x1 = np.array(x[:-1])
    x2 = np.array(x[1:])
    return sum(100 * (x2 - x1**2)**2 + (x1 - 1)**2)


def rastrigin(x):
    return sum(np.square(x) - 10 * np.cos(2 * math.pi * x) + 10)


def griewank(x):
    term1 = 1/4000 * sum(np.square(x - 100))
    term2 = np.prod(np.cos(
               (x - 100) / np.sqrt(range(1, len(x) + 1))))
    return term1 - term2 + 1


def ackley(x):
    n = len(x)
    term1 = -20 * math.exp(-0.2 * np.sqrt(sum(x**2) / n))
    term2 = -math.exp(sum(np.cos(2 * math.pi * x)) / n)
    return term1 + term2 + 20 + math.e


def get_function(function_name):
    if function_name == 'rosenbrock':
        l_bound, u_bound = -5.0, 10.0
        task = 'min'
        return rosenbrock, l_bound, u_bound, task
    elif function_name == 'rastrigin':
        l_bound, u_bound = -5.12, 5.12
        task = 'min'
        return rastrigin, l_bound, u_bound, task
    elif function_name == 'griewank':
        l_bound, u_bound = -10.0, 10.0
        task = 'min'
        return griewank, l_bound, u_bound, task
    elif function_name == 'ackley':
        l_bound, u_bound = -5.0, 5.0
        task = 'min'
        return ackley, l_bound, u_bound, task
    else:
        return None
