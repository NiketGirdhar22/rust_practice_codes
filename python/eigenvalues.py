def transpose(matrix):
    return [list(row) for row in zip(*matrix)]

def dot_product(a, b):
    return sum(x * y for x, y in zip(a, b))

def mat_mul(a, b):
    n = len(a)
    result = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            for k in range(n):
                result[i][j] += a[i][k] * b[k][j]
    return result

def qr_decomposition(a):
    n = len(a)
    q = [[0.0] * n for _ in range(n)]
    r = [[0.0] * n for _ in range(n)]
    a_t = transpose(a)

    for i in range(n):
        v = a_t[i][:]
        for j in range(i):
            dot = dot_product(a_t[i], q[j])
            r[j][i] = dot
            for k in range(n):
                v[k] -= dot * q[j][k]
        norm = sum(x**2 for x in v)**0.5
        for k in range(n):
            q[i][k] = v[k] / norm
        r[i][i] = norm

    return transpose(q), r

def qr_algorithm(a, iterations):
    for _ in range(iterations):
        q, r = qr_decomposition(a)
        a = mat_mul(r, q)
    return [a[i][i] for i in range(len(a))]

n = int(input("Enter matrix size n: "))
matrix = []
print("Enter the matrix elements row by row:")
for i in range(n):
    row = list(map(float, input(f"Row {i+1}: ").strip().split()))
    matrix.append(row)

print("\nMatrix:")
for row in matrix:
    print(row)

eigenvalues = qr_algorithm(matrix, 100)

print("\nApproximated Eigenvalues:")
for i, lam in enumerate(eigenvalues):
    print(f"lambda{i+1} ≈ {lam:.6f}")