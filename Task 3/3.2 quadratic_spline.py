import numpy as np
from scipy.linalg import lu_factor, lu_solve
import matplotlib.pyplot as plt

def f(x):
    return np.exp(-x) * (x**3 - 2*x - 1)

def quadratic(x, a, b, c):
    return a * x**2 + b * x + c

a = 0
b = 5
n = 10

A = np.zeros((n * 3, n * 3))
B = np.zeros(n * 3)

x_nodes = np.linspace(a, b, n + 1)
f_values = f(x_nodes)

x = np.linspace(a, b, 1000)

for i in range(n):
    A[i][3 * i:3 * (i + 1)] = [x_nodes[i]**2, x_nodes[i], 1]
    B[i] = f_values[i]

    A[n + i][3 * i:3 * (i + 1)] = [x_nodes[i + 1]**2, x_nodes[i + 1], 1]
    B[n + i] = f_values[i + 1]

    if i < n-1:
        A[2 * n + i][3 * i:3 * i + 2] = [2 * x_nodes[i + 1], 1]
        A[2 * n + i][3 * (i + 1):3 * (i + 1) + 2] = [-2 * x_nodes[i + 1], -1]

fprime_x0 = (f_values[1] - f_values[0]) / (x_nodes[1] - x_nodes[0])
A[-1][:2] = [2 * x_nodes[0], 1]
B[-1] = fprime_x0
LU, piv = lu_factor(A)
coeffs = lu_solve((LU, piv), B)
spline_coeffs = coeffs.reshape(n, 3)

spline_values = np.zeros_like(x)
for i, coefficient in enumerate(spline_coeffs):
    interval_start = x_nodes[i]
    interval_end = x_nodes[i + 1]
    interval_condition = np.logical_and(x >= interval_start, x <= interval_end)
    a, b, c = coefficient
    spline_values[interval_condition] = quadratic(x[interval_condition], a, b, c)

spline_f_values = np.array(spline_values)
plt.figure(figsize=(10, 8))
plt.plot(x, spline_f_values, label='Quadratic Spline', color='blue')
plt.plot(x, f(x), label="Actual Function", color="red")
plt.scatter(x_nodes, f_values, color='black', marker='o', label='Interpolation Nodes')
plt.title('Quadratic Spline vs. Actual Function')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.legend()
plt.grid(True)
plt.show()
