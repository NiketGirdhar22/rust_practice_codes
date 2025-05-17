def lu_decompose(a):
    n = len(a)
    l = [[0.0] * n for _ in range(n)]
    u = [row[:] for row in a]

    for i in range(n):
        l[i][i] = 1.0

    for i in range(n):
        for j in range(i, n):
            sum_ = sum(l[i][k] * u[k][j] for k in range(i))
            u[i][j] -= sum_

        for j in range(i + 1, n):
            sum_ = sum(l[j][k] * u[k][i] for k in range(i))
            l[j][i] = (u[j][i] - sum_) / u[i][i]
    
    return l, u

def lu_back_substitution(l, u, b):
    n = len(b)
    y = [0.0] * n
    x = [0.0] * n

    for i in range(n):
        y[i] = b[i] - sum(l[i][j] * y[j] for j in range(i))

    for i in range(n - 1, -1, -1):
        x[i] = (y[i] - sum(u[i][j] * x[j] for j in range(i + 1, n))) / u[i][i]

    return x

def round_small_values(matrix, tolerance=1e-6):
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            if abs(matrix[i][j]) < tolerance:
                matrix[i][j] = 0.0

def matrix_multiply(a, b):
    n = len(a)
    result = [[0.0 for _ in range(n)] for _ in range(n)]
    for i in range(n):
        for j in range(n):
            for k in range(n):
                result[i][j] += a[i][k] * b[k][j]
    return result

def print_matrix(label, matrix):
    print(label)
    for row in matrix:
        print(['{:.6f}'.format(x) for x in row])
    print()

size = int(input("Enter the size of the matrix (n x n): "))
print("Enter the elements of the matrix row by row (space-separated):")
matrix = []
for _ in range(size):
    row = list(map(float, input().strip().split()))
    matrix.append(row)

l, u = lu_decompose(matrix)
print_matrix("L matrix:", l)
print_matrix("U matrix:", u)

inverse = [[0.0 for _ in range(size)] for _ in range(size)]
for i in range(size):
    b = [0.0 for _ in range(size)]
    b[i] = 1.0
    col = lu_back_substitution(l, u, b)
    for j in range(size):
        inverse[j][i] = col[j]

round_small_values(inverse)
print_matrix("Inverse matrix:", inverse)

identity = matrix_multiply(matrix, inverse)
round_small_values(identity)
print_matrix("Matrix * Inverse ≈ Identity:", identity)