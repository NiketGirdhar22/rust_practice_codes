def factorial(n):
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result

def exp_approx(x, terms=20):
    total = 0.0
    for k in range(terms):
        total += (x ** k) / factorial(k)
    return total

def compute_second_derivative(n, x, h):
    h_steps = []
    computed = []
    for _ in range(n):
        h_steps.append(h)
        approx = (exp_approx(x + h) - 2 * exp_approx(x) + exp_approx(x - h)) / (h * h)
        computed.append(approx)
        h /= 2
    return h_steps, computed

def write_output(h_steps, computed, x):
    with open("out.dat", "w") as f:
        true_val = exp_approx(x)
        for h, approx in zip(h_steps, computed):
            rel_error = abs(approx - true_val) / true_val
            f.write(f"{h:.6e} {rel_error:.5e}\n")

initial_step = float(input("Initial stepsize: "))
x = float(input("Evaluate at point x: "))
number_of_steps = int(input("Number of steps: "))

h_steps, computed = compute_second_derivative(number_of_steps, x, initial_step)
write_output(h_steps, computed, x)
print("Results written to out.dat")