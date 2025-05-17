import sys
import time

def simple_rand(seed):
    seed = (seed * 1664525 + 1013904223) & 0xFFFFFFFFFFFFFFFF
    rand_val = ((seed >> 16) & 0x7FFF) / 32768.0
    return rand_val, seed

def func(x):
    return 4.0 / (1.0 + x * x)

if len(sys.argv) != 2:
    print(f"Usage: {sys.argv[0]} N")
    sys.exit(1)

try:
    n = int(sys.argv[1])
except ValueError:
    print("Invalid number")
    sys.exit(1)

seed = int(time.time())
sum_fx = 0.0
sum_fx2 = 0.0

for _ in range(n):
    x, seed = simple_rand(seed)
    fx = func(x)
    sum_fx += fx
    sum_fx2 += fx * fx

mean = sum_fx / n
mean2 = sum_fx2 / n
variance = mean2 - mean * mean

print(f"Integral = {mean}, variance = {variance}")