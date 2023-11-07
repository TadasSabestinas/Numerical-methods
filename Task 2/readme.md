# Task 2.1

Solve the nonlinear system of equations AX = F(X) with epsilon accuracy using the iteration method AX^(k+1) = F(X^(k)) and computing the error with the maximum norm. Solve the resulting system of equations in each iteration using LU decomposition. Create the diagonal matrix An*n and the vector function F(X) of length N as follows:

![Salyga](https://i.imgur.com/HjKjFre.png)

Calculate separately: a) the working time of the decomposition of Lu by changing N. b) the time of solving the equations with triangular matrices themselves. Use this to estimate the size complexity of these algorithms.

# Task 2.2

Solve the equation using the method of gradients (maximum slope). After creating a symmetric positive definite matrix A as follows: Generate a symmetric matrix A of size N x N. Recalculate the elements of the main diagonal as follows:

![Salyga2](https://i.imgur.com/JMBhzdk.png) 

This will guarantee positive definiteness. Calculate the error using the maximum norm with epsilon precision. Compare with the running time of solving with the exact method from the first part.

