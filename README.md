# Hamiltonian Neural Networks

<br>

## Background - Physics-Informed Machine Learning (PIML)

Suppose you have a dynamic / physical system to be analyzed / predicted. Several cases:

1) If you don't know the physics -> You cannot use physics much - this may be the scope of Machine Learning - e.g., finding patterns from data, etc.

2) If you know the physics and you can solve the physics equation (e.g., PDE) analytically -> You are good. Analytic math does the job.

3) If you know the physics and you cannot solve the physics equation numerically because of computation limits (e.g., complicated non-linear system) -> Physics-Informed ML (PIML) comes in to play. How? Get the data from the system dynamics -> Train the neural networks, but passing it some hints that are based on known physics.

Even a more funny (shocking) thing is that the neural networks can learn the physics itself - for example, it can learn the conservation law of energy purely from data - `Hamiltonian Neural Networks (HNN)` was born from this context. 

<br>

## Hamiltonian Neural Networks - Time-series predictions for pendulum dynamics.

This work demonstrates `Hamiltonian Neural Networks (HNN)` - to predict (non-linear) pendulum dynamics.

Demo: [https://github.com/uriyeobi/hamlitonian_neural_networks/blob/main/notebooks/pendulum_hnn.ipynb](https://github.com/uriyeobi/hamlitonian_neural_networks/blob/main/notebooks/pendulum_hnn.ipynb)

Blog posts: [Part 1](https://uriyeobi.github.io/2023-05-23/laplace-to-nn-1) | [Part 2](https://uriyeobi.github.io/2023-06-15/laplace-to-nn-2)


<img src="https://github.com/uriyeobi/hamlitonian_neural_networks/blob/main/notebooks/fig/double_pendulum.gif?raw=true" width="500rem">

<img src="https://github.com/uriyeobi/uriyeobi.github.io/blob/main/assets/images/diagram_hnn.png?raw=true" width="800rem">
