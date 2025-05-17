def rk4(f, x0, y0, h):
    k1 = h * f(x0, y0)
    k2 = h * f(x0 + h / 2, y0 + k1 / 2)
    k3 = h * f(x0 + h / 2, y0 + k2 / 2)
    k4 = h * f(x0 + h, y0 + k3)

    print(f"k1 = {k1}")
    print(f"k2 = {k2}")
    print(f"k3 = {k3}")
    print(f"k4 = {k4}")

    return y0 + (k1 + 2*k2 + 2*k3 + k4) / 6

func_str = input("Enter the differential equation f(x, y): ").strip()

allowed_names = {
    "x": 0,
    "y": 0,
    "sin": __import__("math").sin,
    "cos": __import__("math").cos,
    "exp": __import__("math").exp,
    "log": __import__("math").log,
    "sqrt": __import__("math").sqrt,
    "pow": pow,
}

def f(x, y):
    local_dict = {"x": x, "y": y}
    return eval(func_str, {"__builtins__": None}, {**allowed_names, **local_dict})

x0 = float(input("Enter initial x (x0): "))
y0 = float(input("Enter initial y (y0): "))
h = float(input("Enter step size (h): "))

y1 = rk4(f, x0, y0, h)
print(f"Estimated y1 = {y1}")