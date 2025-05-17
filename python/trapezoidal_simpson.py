def evaluate_function(expr, x):
    allowed_names = {'x': x, '__builtins__': None}
    return eval(expr, {"__builtins__": None}, allowed_names)

def trapezoidal_rule(expr, a, b, n):
    h = (b - a) / n
    total = 0.5 * (evaluate_function(expr, a) + evaluate_function(expr, b))
    for i in range(1, n):
        x = a + i * h
        total += evaluate_function(expr, x)
    return total * h

def simpson_rule(expr, a, b, n):
    if n % 2 != 0:
        raise ValueError("Simpson's rule requires an even number of intervals.")
    h = (b - a) / n
    total = evaluate_function(expr, a) + evaluate_function(expr, b)
    for i in range(1, n):
        x = a + i * h
        coeff = 4 if i % 2 != 0 else 2
        total += coeff * evaluate_function(expr, x)
    return total * h / 3

print("Enter the function f(x) using Python syntax (e.g., x**2 + 1):")
expr = input().strip()

a = float(input("Enter lower limit a: "))
b = float(input("Enter upper limit b: "))
n = int(input("Enter number of intervals n: "))

trapezoidal_result = trapezoidal_rule(expr, a, b, n)
print(f"Result using Trapezoidal Rule: {trapezoidal_result}")

if n % 2 != 0:
    print("Simpson's rule requires an even number of intervals. Skipping Simpson's Rule.")
else:
    simpson_result = simpson_rule(expr, a, b, n)
    print(f"Result using Simpson's Rule: {simpson_result}")
