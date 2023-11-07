import numpy as np
import time
import matplotlib.pyplot as plt

def lu_function(n):
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

    # tiesine eiga
    def triangular_matrix_substitution(LU, B, n, is_lower=True):
        X = np.zeros(n)

        if is_lower:
            for i in range(n):
                X[i] = B[i]
                for j in range(i):
                    X[i] -= LU[i][j] * X[j]
                X[i] /= LU[i][i]
        else:
            for i in range(n - 1, -1, -1):
                X[i] = B[i]
                for j in range(i + 1, n):
                    X[i] -= LU[i][j] * X[j]
                X[i] /= LU[i][i]

        return X

    c = 1 / ((n + 1) ** 2)
    A = np.diag(2 * np.ones(n)) - np.diag(np.ones(n - 1), -1) - np.diag(np.ones(n - 1), 1)
    # print("Matrix A:\n", A)

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
        # Y = np.linalg.solve(L, F)
        # X_new = np.linalg.solve(U, Y)
        end_time = time.time()
        elapsed_time = end_time - start_time
        

        error = np.max(np.abs(X_new - X))
        if error < epsilon:
            break
        X = X_new

    # print("Solution LU X:", X)

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
        
        F = np.random.rand(N)
        start_time = time.time()
        D = triangular_matrix_substitution(L, F, N, is_lower=True)
        X_new = triangular_matrix_substitution(U, D, N, is_lower=False)
        end_time = time.time()
        elapsed_time = end_time - start_time
        runtimes_matrix.append(elapsed_time)
    
    print(f"When N={n}, the triangular_matrix_substitution runtime is: {elapsed_time:.15f} seconds")
    print(f"When N={n}, The LU algorithm run time is: {elapsed_time_n:.20f}")

def gradient_function(n):
    def generate_matrix(n):
        A = np.diag(2 * np.ones(n)) - np.diag(np.ones(n - 1), -1) - np.diag(np.ones(n - 1), 1) 
        A = (A + A.T) / 2  
        for i in range(n):
            A[i, i] = np.sum(np.abs(A[i, :])) 
        return A

    A = generate_matrix(n)
    

    # 1) choose x0
    X = np.zeros(n)
    B = np.random.rand(n)  
    epsilon = 1e-3  

    # 2) Calculate residual pk, pK = axK - B 
    pk = B - np.dot(A, X)
    previous_X = X.copy()
    delta_x = 0

    # 5) check for convergence ||xk+1 - x|| < epsilon
    while delta_x < epsilon:
        RR = 0
        AR = np.zeros_like(pk)
        start_time = time.time()
        for i in range(n):
            RR += pk[i] * pk[i]
            for j in range(n):
                AR[i] += A[i][j] * pk[j]
        # 3) Calculate alpha
        alpha = RR / np.sum(AR * pk)
    
        for i in range(n):
            # 4) update xk+1 = xk + alpha*pk
            X[i] += alpha * pk[i]

        end_time = time.time()
        delta_x = np.max(np.abs(X - previous_X))
        
        previous_X = X.copy()
        pk = B - np.dot(A, X)

    # print(f"Solution X:")
    # print(X)
    print(f"Gradient Descent Time: {end_time - start_time:.15f} seconds")

n = int(input("Enter N: \n"))
lu_function(n)
gradient_function(n)