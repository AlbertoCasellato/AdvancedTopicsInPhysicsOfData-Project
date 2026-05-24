# Solar Dynamo ENCA

This project applies the **Explicit Noise Conditional Autoencoder (ENCA)** 
to the solar dynamo model — a stochastic delay differential equation (SDDE) 
that simulates sunspot activity.

The goal is to learn minimal and informative summary statistics from 
simulated solar dynamo time series, which can then be used for 
likelihood-free Bayesian inference of the model parameters.

The model is simulated in Julia and the autoencoder is implemented in PyTorch.