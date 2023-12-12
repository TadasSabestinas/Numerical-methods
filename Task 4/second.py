import numpy as np
from scipy.integrate import quad

def f(x):
    return x**2 + np.sqrt(x)

def right_rectangles_method(a, b, n):
    h = (b - a) / n
    x_values = np.linspace(a, b, n, endpoint=False) + h 
    integral_approx = np.sum(f(x_values)) * h
    return integral_approx

def derivative_f(x):
    return 2 * x + 1 / (2 * np.sqrt(x))

a = 1
b = 4

n_values = [5, 10, 20, 40, 80, 160, 320, 640]

true_value, _ = quad(f, a, b)

approx_error_values = []
true_error_values = []

print(f"| {'n':<2} | {'h':<10} | {'Approximate Integral':<20} | {'True Error':<20} | {'Error of approximation':<20} | {'True error and Approximate error ratio':<20} |")

for n in n_values:
    h = (b - a) / n
    integral_approx = right_rectangles_method(a, b, n)

    x_values = np.linspace(a, b, n, endpoint=False) + h
    M_i = np.max(derivative_f(x_values))

    true_error = np.abs(true_value - integral_approx)
    true_error_values.append(true_error)

    approx_error = M_i * ((b-a)/2) * h
    approx_error_values.append(approx_error)

    if (true_error <= approx_error):
        print(f"| {n:<2} | {h:<10.6f} | {integral_approx:<10.6f}          | {true_error:<20.6f} | {approx_error:<20.6f} | {true_error/approx_error:<20.6f}                |")

print(f"True integral value: {true_value:<20.6f}")

# print(f" {'True error ratio:':<20}| {'Approximate error ratio:':<20}")
# for n in range(len(n_values) - 1): 
#     print(f" {true_error_values[n]/true_error_values[n+1]} | {approx_error_values[n]/approx_error_values[n+1]}")
