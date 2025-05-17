import math

def compute_second_derivative(n, x, h):
    h_steps = []
    computed = []
    for _ in range(n):
        h_steps.append(h)
        approx = (math.exp(x + h) - 2 * math.exp(x) + math.exp(x - h)) / (h * h)
        computed.append(approx)
        h /= 2
    return h_steps, computed

def write_output(h_steps, computed, x):
    with open("out.dat", "w") as f:
        for h, approx in zip(h_steps, computed):
            true_val = math.exp(x)
            rel_error = abs(approx - true_val) / true_val
            f.write(f"{math.log10(h):.6f} {math.log10(rel_error):12.5e}\n")

initial_step = float(input("Initial stepsize: "))
x = float(input("Evaluate at point x: "))
number_of_steps = int(input("Number of steps: "))

h_steps, computed = compute_second_derivative(number_of_steps, x, initial_step)
write_output(h_steps, computed, x)
print("Results written to out.dat")