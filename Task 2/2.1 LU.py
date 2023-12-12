import numpy as np
import time
import matplotlib.pyplot as plt

def lu_decomposition(A, n):
    L = np.zeros((n, n))
    U = np.zeros((n, n))

    for i in range(n):
        L[i][i] = 1

        for j in range(i, n):
            U[i][j] = A[i][j]
            for k in range(i):
                U[i][j] -= L[i][k] * U[k][j]

        for j in range(i + 1, n):
            L[j][i] = A[j][i]
            for k in range(i):
                L[j][i] -= L[j][k] * U[k][i]
            L[j][i] /= U[i][i]

    return L, U

def triangular_matrix_substitution(LU, B, n, is_lower=True):
    X = np.zeros(n)

    if is_lower:
        # forward substitution
        for i in range(n):
            X[i] = B[i]
            for j in range(i):
                X[i] -= LU[i][j] * X[j]
            X[i] /= LU[i][i]
    else:
        # backward substitution
        for i in range(n - 1, -1, -1):
            X[i] = B[i]
            for j in range(i + 1, n):
                X[i] -= LU[i][j] * X[j]
            X[i] /= LU[i][i]

    return X

n = int(input("Enter N: \n"))
c = 1 / ((n + 1) ** 2)
A = np.diag(2 * np.ones(n)) - np.diag(np.ones(n - 1), -1) - np.diag(np.ones(n - 1), 1)
print("Matrix A:\n", A)

X = np.zeros(n)
epsilon = 1e-3

L, U = lu_decomposition(A, n)

while True:
    F = np.zeros(n)
    for i in range(n):
        if i == 0:
            F[i] = c + 2 * (X[i + 1] - 0) ** 2
        elif i > 0 and i < n - 1:
            F[i] = c + 2 * (X[i + 1] - X[i - 1]) ** 2
        else:
            F[i] = c + 2 * (0 - X[n-1]) ** 2
    # print("F(X):", F)

    start_time = time.time()
    d = triangular_matrix_substitution(L, F, n, is_lower=True)
    X_new = triangular_matrix_substitution(U, d, n, is_lower=False)
    end_time = time.time()
    elapsed_time = end_time - start_time
    

    error = np.max(np.abs(X_new - X))
    if error < epsilon:
        break
    X = X_new

print("Solution X:", X)

N_values = np.arange(5, n+1, 5)
runtimes_matrix = []
runtimes = []

for N in N_values:
    Atime = np.diag(2 * np.ones(N)) - np.diag(np.ones(N - 1), -1) - np.diag(np.ones(N - 1), 1)
    start_time_n = time.time()
    lu_decomposition(Atime, N)
    end_time_n = time.time()
    elapsed_time_n = end_time_n - start_time_n
    runtimes.append(elapsed_time_n)
    print(f"When N={N}, The LU algorithm run time is: {elapsed_time_n:.20f}")
    
    F = np.random.rand(N) 

    start_time = time.time()
    D = triangular_matrix_substitution(L, F, N, is_lower=True)
    X_new = triangular_matrix_substitution(U, D, N, is_lower=False)
    end_time = time.time()
    elapsed_time = end_time - start_time
    runtimes_matrix.append(elapsed_time)

    print(f"When N={N}, the triangular_matrix_substitution runtime is: {elapsed_time:.15f} seconds")

n_double = 2 * n
Atime_double = np.diag(2 * np.ones(n_double)) - np.diag(np.ones(n_double - 1), -1) - np.diag(np.ones(n_double - 1), 1)
start_time_double = time.time()
L, U = lu_decomposition(Atime_double, n_double)
end_time_double = time.time()
elapsed_time_double = end_time_double - start_time_double

print(f"The LU algorithm time when n = {n} is: ", elapsed_time_n)
print(f"The LU algorithm run time when n = {n_double} is: ", elapsed_time_double)
print(f"The division between n = {n_double} and n = {n} is: ", (elapsed_time_double/elapsed_time_n))

F = np.random.rand(n_double)

start_time_double_matrix = time.time()
D = triangular_matrix_substitution(L, F, n_double, is_lower=True)
X_new = triangular_matrix_substitution(U, D, n_double, is_lower=False)
end_time_double_matrix = time.time()
elapsed_time_double_matrix = end_time_double_matrix - start_time_double_matrix

print(f"The Triangular matrix substitution algorithm run time when n = {n} is: ", elapsed_time)
print(f"The Triangular matrix substitution algorithm run time when n = {n_double} is: ", elapsed_time_double_matrix)
print(f"The division between n = {n_double} and n = {n} is: ", (elapsed_time_double_matrix/elapsed_time))
