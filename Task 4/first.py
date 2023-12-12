import numpy as np
import matplotlib.pyplot as plt

x_values = []
y_values = []

with open('duom2.txt') as file:
    lines = file.readlines()
    for line in lines:
        data_point = line.split()
        x_values.append(float(data_point[0]))
        y_values.append(float(data_point[1]))

x_values = np.array(x_values)
y_values = np.array(y_values)

# linear Function (a0 + a1x)
phi_linear = np.column_stack((np.ones_like(x_values), x_values)) 
phi_linear_transpose = phi_linear.T
phi_linear_product = np.dot(phi_linear_transpose, phi_linear) # multiplies matrices phi ir phi^T
phi_linear_inverse = np.linalg.inv(phi_linear_product) # calculates inverse matrix (phi^T * phi)^-1
phi_linear_product_vector = np.dot(phi_linear_transpose, y_values) # calculates right-hand side of the equation phi^T * y
coefficients_linear = np.dot(phi_linear_inverse, phi_linear_product_vector) #finds coefficients = (phi^T * phi)^-1 * phi^T * y

# quadratic Function (a0 + a1x + a2x^2)
phi_quadratic = np.column_stack((np.ones_like(x_values), x_values, x_values**2))
phi_quadratic_transpose = phi_quadratic.T
phi_quadratic_product = np.dot(phi_quadratic_transpose, phi_quadratic)
phi_quadratic_inverse = np.linalg.inv(phi_quadratic_product)
phi_quadratic_product_vector = np.dot(phi_quadratic_transpose, y_values)
coefficients_quadratic = np.dot(phi_quadratic_inverse, phi_quadratic_product_vector)

# calculates approximated values
y_linear_approx = np.dot(phi_linear, coefficients_linear)
y_quadratic_approx = np.dot(phi_quadratic, coefficients_quadratic)

sse_linear = np.sum((y_values - y_linear_approx)**2)
sse_quadratic = np.sum((y_values - y_quadratic_approx)**2)

X = np.linspace(0, 5, 1000)

y_linear = coefficients_linear[0] + coefficients_linear[1] * X
y_quadratic = coefficients_quadratic[0] + coefficients_quadratic[1] * X + coefficients_quadratic[2] * X**2

if sse_linear < sse_quadratic:
    print(f"Linear model is a better fit. ({sse_linear})")
else:
    print(f"Quadratic model is a better fit. ({sse_quadratic})")

equation_linear = f'Linear Function: y = {coefficients_linear[0]:.2f} + {coefficients_linear[1]:.2f}x'
equation_quadratic = f'Quadratic Function: y = {coefficients_quadratic[0]:.2f} + {coefficients_quadratic[1]:.2f}x + {coefficients_quadratic[2]:.2f}x^2'

plt.plot(X, y_linear, color="green", label=equation_linear)
plt.plot(X, y_quadratic, color="purple", label=equation_quadratic)
plt.scatter(x_values, y_values, color="grey", label='Original Data')
plt.legend()
plt.show()
