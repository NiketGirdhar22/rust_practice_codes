def func(x):
    return x**3 - x - 2

def func_derivative(x):
    return 3 * x**2 - 1

def bisection(f, a, b, tol, nmax):
    for n in range(1, nmax + 1):
        c = (a + b) / 2.0
        print(f"n={n}\ta={a:.6f}\tb={b:.6f}\tc={c:.6f}\tf(c)={f(c):.6f}")
        if abs(f(c)) < tol or abs(b - a) / 2.0 < tol:
            return c
        if f(c) * f(a) > 0:
            a = c
        else:
            b = c
    return None

def secant(f, x0, x1, tol, nmax):
    for n in range(1, nmax + 1):
        f_x0 = f(x0)
        f_x1 = f(x1)
        if abs(f_x1 - f_x0) < tol:
            return None
        x2 = x1 - f_x1 * (x1 - x0) / (f_x1 - f_x0)
        print(f"n={n}\tx0={x0:.6f}\tx1={x1:.6f}\tx2={x2:.6f}\tf(x2)={f(x2):.6f}")
        if abs(x2 - x1) < tol:
            return x2
        x0 = x1
        x1 = x2
    return None

def newton_raphson(f, f_prime, x0, tol, nmax):
    for n in range(1, nmax + 1):
        fx = f(x0)
        fpx = f_prime(x0)
        if abs(fpx) < tol:
            return None
        x1 = x0 - fx / fpx
        print(f"n={n}\tx0={x0:.6f}\tx1={x1:.6f}\tf(x1)={f(x1):.6f}")
        if abs(x1 - x0) < tol:
            return x1
        x0 = x1
    return None

tol = 0.001
nmax = 100

print("Bisection Method")
root_bis = bisection(func, 1.0, 2.0, tol, nmax)
if root_bis is not None:
    print(f"Root found by Bisection: {root_bis:.6f}")
else:
    print("Bisection failed to converge")

print("\nSecant Method")
root_sec = secant(func, 1.0, 2.0, tol, nmax)
if root_sec is not None:
    print(f"Root found by Secant: {root_sec:.6f}")
else:
    print("Secant failed to converge")

print("\nNewton-Raphson Method")
root_newton = newton_raphson(func, func_derivative, 1.0, tol, nmax)
if root_newton is not None:
    print(f"Root found by Newton-Raphson: {root_newton:.6f}")
else:
    print("Newton-Raphson failed to converge")