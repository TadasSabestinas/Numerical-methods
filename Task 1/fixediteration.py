import numpy as np
import matplotlib.pyplot as plt

# function g(x) = e^(-x)
def g(x):
    return np.exp(-x)

def gprime(x):
    return -np.exp(-x)


def fixed_point_iteration(a, b, x, max_iterations, epsilon):
    
    q = np.max(np.abs(-np.exp(-np.linspace(a, b, 1000))))

    if (gprime(x) <= q < 1):
        iterations = []

        for i in range(max_iterations):
            gx = g(x)
            iterations.append([i + 1, x, gx, abs(gx - x)])


            if abs(gx - x) <= ((1-q)/q)*epsilon:
                break

            x = gx

        return gx, iterations
    else:
        return False, False

max_iterations = 100
epsilon = 1e-4

# 0.5 2 1
a = float(input("Enter the start of the interval: \n"))
b = float(input("Enter the end of the interval: \n"))
x = float(input("Enter your initial guess \n"))
initial_guess = x
while True:
    if (a < b and a < x < b):
        root, iteration_data = fixed_point_iteration(a, b, x, max_iterations, epsilon)

        if(root == False and iteration_data == False):
            print("q is higher than 1, select a suitable interval")
            break
        else:
            gx_values = [data[2] for data in iteration_data]


            print("\nIterations:")
            print("Iteration |   x   |   g(x)   |   Error")
            print("---------------------------------------------" )
            for data in iteration_data:
                iteration, x, gx, error = data
                print(f"{iteration:^5} | {x:.8f} | {gx:.8f} | {error:.6e}")

            print(f"Root: {root:.8f}")

            x_values = np.arange(1, len(iteration_data) + 1)
            errors = [data[3] for data in iteration_data]
            plt.figure(figsize=(12, 6))
            # plt.xlim(0.525, 0.6)
            x_values = np.linspace(a-4, b, 1000)
            y_values = g(x_values)
            plt.plot(x_values, x_values, label="y = x", color='black')
            plt.plot(x_values, y_values, label="g(x) = e^(-x)")
            plt.scatter(initial_guess, gx_values[0], color="purple", marker="o", label="Initial guess", zorder=3)
            plt.scatter(gx_values, gx_values, color='purple', marker='o', label='Iteration points', zorder=4)
            plt.scatter(root, root, color='green', marker='o', label='Root', zorder=5) 
            plt.xlabel("x")
            plt.ylabel("g(x)")
            plt.axhline(0, color='red', label="y = 0")
            plt.grid(True, linestyle='--')
            plt.legend()
            plt.title("Fixed-Point Iteration for g(x) = e^(-x)")

            plt.show()
            break
    else:
        print("Invalid interval or initial guess. Please enter again")
        a = float(input("Enter the start of the interval: \n"))
        b = float(input("Enter the end of the interval: \n"))
        x = float(input("Enter your initial guess \n"))
        initial_guess = x