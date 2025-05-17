import csv

PI = 3.141592653589793
PI4 = 4 * PI * PI

def solver(m, dt, t0):
    num_intervals = round(m / dt)
    print(f"Time steps: {num_intervals}")

    t = [t0]
    x = [1.0]
    y = [0.0]
    vx = [2.0 * PI]
    vy = [0.0]
    r = [(x[0]**2 + y[0]**2)**0.5]
    v = [(vx[0]**2 + vy[0]**2)**0.5]

    for n in range(num_intervals):
        tn = t0 + (n + 1) * dt
        t.append(tn)

        xn = x[n] + dt * vx[n]
        yn = y[n] + dt * vy[n]
        x.append(xn)
        y.append(yn)

        r3 = (xn**2 + yn**2)**1.5
        vxn = vx[n] - dt * PI4 * xn / r3
        vyn = vy[n] - dt * PI4 * yn / r3
        vx.append(vxn)
        vy.append(vyn)

        v.append((vxn**2 + vyn**2)**0.5)
        r.append((xn**2 + yn**2)**0.5)

    return r, v, t

def save_to_csv(filename, t, r, v):
    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['time', 'radius', 'speed'])
        for i in range(len(t)):
            writer.writerow([f"{t[i]:.5f}", f"{r[i]:.5f}", f"{v[i]:.5f}"])

m = 20.0
dt = 0.01
t0 = 0.0

r, v, t = solver(m, dt, t0)

try:
    save_to_csv("orbit.csv", t, r, v)
    print("Simulation complete. Output written to orbit.csv")
except IOError as e:
    print(f"Failed to save data: {e}")