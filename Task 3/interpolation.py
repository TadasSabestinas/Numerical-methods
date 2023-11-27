import numpy as np
import scipy.interpolate as interpolate
import matplotlib.pyplot as plt

def f(x):
    return np.exp(-x) * (x**3 - 2*x - 1)

def divided_difference(x_values, f_values, order):
    if order == 0:
        return f_values
    else:
        return (divided_difference(x_values[1:], f_values[1:], order-1) - divided_difference(x_values[:-1], f_values[:-1], order-1)) / (x_values[order:] - x_values[:-order])

def interpolation_polynomial(x, x_values, f_values, order):
    result = 0
    for i in range(order+1):
        newton_term = divided_difference(x_values[:i+1], f_values[:i+1], i)
        for j in range(i):
            newton_term *= (x - x_values[j])
        result += newton_term
    return result

def interpolated_value_at5(x):
    for i in range(len(x_values)-6, -1, -1):
        if x_values[i] <= x <= x_values[i+5]:
            return interpolation_polynomial(x, x_values[i:i+6], y_values[i:i+6], 5)

user_input = float(input("Enter a number between 0 and 5: "))
a = 0
b = 5
n = 6

x_values = np.linspace(a, b, n)
y_values = f(x_values)

error = np.abs(f(user_input) - interpolated_value_at5(user_input))
print(f"Original function's value when x = {user_input}: [{f(user_input)}]")
print(f"5th order interpolation polynomial's value when x = {user_input}: {interpolated_value_at5(user_input)}")
print(f"Error at x = {user_input}: {error}")

x = np.linspace(0, 5, 1000)
y = f(x)

plt.figure(figsize=(12, 8))

plt.plot(x, y, label='True function')
plt.plot(x, np.vectorize(interpolated_value_at5)(x), label='Interpolation polynomial', linestyle='--')
plt.scatter(x_values, y_values, label='Function Nodes')
plt.scatter(user_input, f(user_input), color='red', marker='o', label='User Input')
plt.legend()
plt.grid(True)
plt.show()
