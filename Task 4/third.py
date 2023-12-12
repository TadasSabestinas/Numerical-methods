import numpy as np

def f(x):
    return x**2 + np.sqrt(x)

def integrate_subinterval(f, a, b):
    h = (b - a) / 2
    result = h * (f(a) + f(b))
    return result

def adaptive_right_rectangles_integration_recursive(f, a, b, epsilon):
    def recursive_integration(a, b):
        mid = (a + b) / 2
        left_integral = integrate_subinterval(f, a, mid)
        right_integral = integrate_subinterval(f, mid, b)
        total_integral = left_integral + right_integral

        if np.abs(left_integral - right_integral) < epsilon:
            return total_integral
        else:
            left_recursive = recursive_integration(a, mid)
            right_recursive = recursive_integration(mid, b)
            return left_recursive + right_recursive

    integral_approximation = recursive_integration(a, b)
    return integral_approximation

a = 1
b = 4
epsilon = 0.01

result = adaptive_right_rectangles_integration_recursive(f, a, b, epsilon)

print("Approximal Integral:", result)
