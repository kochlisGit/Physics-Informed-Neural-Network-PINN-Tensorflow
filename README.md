# Physics-Informed-Neural-Network-PINN-
Implementation of a Physics Informed Neural Network (PINN) written in Tensorflow v2, which is capable of solving Partial Differential Equations.

# Description

Neural networks are machine learning models, which try to map an input `X` and to an output `Y`. These models are widely used in Supervised Problems, where there are datasets which contain both the inputs and outputs. However, when modeling PDEs or ODEs, the solution y is unknown, so we cannot directly train a neural network to map the inputs x to the correct outputs. However, we can re-write the equations so that they equal to zero and use these equations as loss functions. Since the goal of a neural network is to minimize its loss function, then once it minimizes the above loss function, then the model will subsequently learn to solve the equation. This method is both quite accurate & fast. 

For example, let's consider the following equation:

`dy/dx = y with y(0) = 1 from x = 0 to 2.`

Obviously, the solution of the above equation is y(x) = e^x. However, we are going to train a neural network to estimate the solution of the above equation. The equation can be also written as:

`dy/dx - y = 0`

So we have 2 losses here:

1. PDE Loss = `MSE(dy/dx - y) = (dy/dx - y)^2`, which should be minimized to zero.
2. IC Loss = `MSE(y(0) - 1) = (y(0) - 1)^2`

So, the loss of our neural network will be:

`Total Loss = PDE Loss + IC Loss = (dy/dx - y)^2 + (y(0) - 1)^2`

Now, we can tell the optimizer of our network to minimize the `Total Loss` to zero and solve our system. The coolest part about neural networks is that they can compute the derivative of y with respect to x via **Back-Propagation**.

# Architecture

A typical PINN architecture can be visualized as follows:

![PINN Image](https://github.com/kochlisGit/Physics-Informed-Neural-Network-PINN-/blob/main/pinn_arch.png)

The training data are passed into the neural network and y = NN(x) is computed. Then, we compute the loss of the PDE, as well the losses of the initial / boundary conditions. Then, we train the neural network using as loss:

`Total Loss = PDE Loss + BC Loss + IC Loss`

Any loss function can be used for computing the losses (e.g. `MSE, MAE`, etc.)

# Paper
https://www.cs.uoi.gr/~lagaris/papers/TNN-LLF.pdf

In this work, the authors present PINNs, a Neural Network architecture capable of solving Partial Differential Equations (PDEs), as well as coupled Ordinary Differential Equations (ODEs).


