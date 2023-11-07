# Task 2.1

Program that solves the nonlinear system of equations AX = F(X) with ϵ accuracy using the iteration method AX^(k+1) = F(X^(k)) and computes the error with the maximum norm. Solves the resulting system of equations in each iteration using LU decomposition. Creates the diagonal matrix An*n and the vector function F(X) of length N as follows:

![Salyga](https://i.imgur.com/HjKjFre.png)

Calculates separately: a) the working time of the decomposition of Lu by changing N. b) the time of solving the equations with triangular matrices themselves. Used this to estimate the size complexity of these algorithms.

# Task 2.2

Solved the equation using the method of gradients (maximum slope). After creating a symmetric positive definite matrix A as follows: Generated a symmetric matrix A of size N x N. Recalculated the elements of the main diagonal as follows:

![Salyga2](https://i.imgur.com/JMBhzdk.png) 

This guaranteed positive definiteness. Calculated the error using the maximum norm with ϵ precision. Compared with the running time of solving with the exact method from the first part.

