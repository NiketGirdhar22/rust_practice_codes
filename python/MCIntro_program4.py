import sys
import math

def simple_rand(seed):
    seed[0] = (seed[0] * 1664525 + 1013904223) & 0xFFFFFFFFFFFFFFFF
    return ((seed[0] >> 16) & 0x7FFF) / 32768.0

def integrand(x):
    a = 1.0
    b = 0.5
    x2 = sum(xi * xi for xi in x)
    xy = (x[0] - x[3])**2 + (x[1] - x[4])**2 + (x[2] - x[5])**2
    return math.exp(-a * x2 - b * xy)

def montecarlo_integration(samples, l, jacobi):
    sum_fx = 0.0
    sum_fx2 = 0.0
    seed = [12345]
    for _ in range(samples):
        x = [ -l + 2 * l * simple_rand(seed) for _ in range(6) ]
        fx = integrand(x)
        sum_fx += fx
        sum_fx2 += fx * fx
    mean = sum_fx / samples
    mean2 = sum_fx2 / samples
    integral = jacobi * mean
    sigma = jacobi * math.sqrt((mean2 - mean * mean) / samples)
    return integral, sigma

if len(sys.argv) != 2:
    print(f"Usage: {sys.argv[0]} number_of_samples")
    sys.exit(1)
n = int(sys.argv[1])
l = 5.0
jacobi = (2.0 * l)**6
integral, sigma = montecarlo_integration(n, l, jacobi)
print(f"Monte Carlo result = {integral:10.8E}")
print(f"Sigma             = {sigma:10.8E}")