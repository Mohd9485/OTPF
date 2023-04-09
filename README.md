# Optimal Transport Particle Filters (OTPFs)

This repository is by Mohammad Al-Jarrah, [Bamdad Hosseini](https://bamdadhosseini.org/), [Amirhossein Taghvaei](https://www.aa.washington.edu/facultyfinder/amir-taghvaei) and contains the Pytorch source code to reproduce the experiments in our 2023 paper [Optimal Transport Particle Filters](https://arxiv.org/abs/2304.00392).

To illustrate the proposed OTPF in comparison with two other filters: the Ensemble Kalman Filter (EnKF),
and the sequential importance resampling (SIR) PF, we are using the following model:

$$
\begin{aligned}
    X_{t} &= (1-\alpha) X_{t-1} + 2\sigma V_t,\quad X_0 \sim \mathcal{N}(0,I_n)\\
    Y_t &= h(X_t) + \sigma W_t
\end{aligned}
$$

for $t=1,2,3,\ldots$, where $X_t,Y_t \in \mathbb R^n$, $\{V_t\}_{t=1}^\infty$ and $\{W_t\}_{t=1}^\infty$ are i.i.d sequences of $n$-dimensional standard Gaussian random variables, $\alpha=0.1$ and $\sigma=\sqrt{0.1}$. We use three observation functions:

$$
\begin{aligned}
    h(x)=x,\quad h(x)=x \odot x,\quad h(x)=x \odot x \odot x
\end{aligned}
$$

where $\odot$ denotes the element-wise (i.e., Hadamard) product when $x$ is a vector.

<p align="center">
<img src="/images/X.png" width="250" height="250"><img src="/images/XX.png" width="250" height="250"><img src="/images/XXX.png" width="250" height="250">
</p>
<p align="center">
<img src="/images/mse_X.png" width="250" height="250"><img src="/images/mse_XX.png" width="250" height="250"><img src="/images/mse_XXX.png" width="250" height="250">
</p>

Please consider citing our paper if you find this repository useful for your publication.

```
@article{al2023optimal,
  title={Optimal Transport Particle Filters},
  author={Al-Jarrah, Mohammad and Hosseini, Bamdad and Taghvaei, Amirhossein},
  journal={arXiv preprint arXiv:2304.00392},
  year={2023}
}
```

## Setup
* Python/Numpy
* PyTorch

## Running the code and Regenerating data and figures.
1. Run the 'main.py' file to regenerate and save the date. There are multiple things you can change in the code:
  - The observation function 'h(x)', please use the desired observation function here.
  - The number of simulations 'AVG_SIM', we used 100 simulations in our paper, but you can change that to a smaller number to get faster results.
  - The number final number of iterations 'parameters['Final_Number_ITERATION']'.
  - Other parameters to choose from like the noise level, the number of particles 'J',..., etc.
2. Use the file 'import_DATA.py' to import and plot all the desired figures. Note here that we will plot the 'mse' for both $\phi(X)=X$ and $\phi(X)=max(0,X)$.

Note: Unfortunately, we ran a random seed every time we ran the code, so we do not have a seed function to provide identical results to our paper, but the figure should be close enough.

