import numpy as np
import matplotlib.pyplot as plt

#function f(x) = e^(-x) - x
def f(x):
    return np.exp(-x) - x

def bisection(a, b, max_iterations):

    if f(a) * f(b) >= 0:
        raise ValueError("Function values at endpoints must have opposite signs.")

    iterations = []
    
    for i in range(max_iterations):
        c = (a + b) / 2.0
        f_c = f(c)
        epsilon = abs(b - a)
        
        iterations.append([i + 1, a, b, c, f_c, epsilon])
        
        if f_c == 0.0 or epsilon < 1e-4:
            break
        elif f(a) * f_c < 0:
            b = c
        else:
            a = c
    
    return c, iterations

def plot_iteration(iteration_number, a, b):

    x_values = np.linspace(a, b, 100) 
    y_values = np.exp(-x_values) - x_values
    plt.plot(x_values, y_values, label="e^(-x) - x")

    selected_iteration = iteration_data[iteration_number - 1]
    a_selected = selected_iteration[1]
    b_selected = selected_iteration[2]
    c_selected = selected_iteration[3]

    plt.scatter(c_selected, 0, color='blue', marker='o', label=f'Iteration {iteration_number}')
    plt.axvline(a_selected, color='purple', linestyle='--', linewidth=0.8)
    plt.axvline(b_selected, color='purple', linestyle='--', linewidth=0.8)


    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.axhline(0, color='red', linewidth=0.8, label="y = 0")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.title(f"Graph of f(x) = e^(-x) - x for Iteration {iteration_number}")

    plt.draw() 

max_iterations = 100
a = float(input("Enter the start of the interval: \n"))
b = float(input("Enter the end of the interval: \n"))

a_graph = a
b_graph = b

root, iteration_data = bisection(a, b, max_iterations)
print("\nIterations:")
print("Iteration |   a    |            b    |       c   |  f(c)      |  Error")
print("------------------------------------------------------------------------------" )
for data in iteration_data:
    iteration, a, b, c, f_c, epsilon = data
    print(f"{iteration:^10} | {a:.8f} | {b:.8f} | {c:.8f} | {f_c:.8f} | {epsilon:.6e}")
    if(abs(b-a) < 1e-4):
        print(f"Solution found, root is {c:.8f}")

iteration_amount = np.arange(1, len(iteration_data) + 1)
errors = [data[5] for data in iteration_data]

plt.figure(figsize=(12,6))

c_values = [data[3] for data in iteration_data]

x_values = np.linspace(a_graph, b_graph, 100)
y_values = np.exp(-x_values) - x_values
plt.plot(x_values, y_values, label="e^(-x) - x")
plt.scatter(c_values[:-1], [0] * len(c_values[:-1]), color='blue', marker='o', label='Iteration Points')
plt.scatter(c_values[-1], 0, color='green', marker='o', label='Root', zorder=5) # zorder makes the green dot appear on top of everything
plt.xlabel("x")
plt.ylabel("f(x)")
plt.axhline(0, color='red', linewidth=0.8, label="y = 0")
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.title("Graph of f(x) = e^(-x) - x with Iteration Points")
plt.show()


iteration_selector = int(input("Enter the iteration number to visualize:"))

while True:
    if 1 <= iteration_selector <= len(iteration_data):
        a = a_graph
        b = b_graph
        plot_iteration(iteration_selector, a, b)
        plt.show()
        break
    else:
        print("Invalid iteration number. Please select a valid iteration.")
        iteration_selector = int(input("Enter the iteration number to visualize:"))
