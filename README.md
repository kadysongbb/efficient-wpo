## Dristributionally Robust Trust Region Policy Optimization 

### Toy Version: 
* Two algorithms implemented: DR TRPO KL, DR TRPO Wasserstein
* This version can run on OpenAI Gym toy text environment, i.e. Taxi-v3, Nchain-v0.
* Modified based on https://github.com/pat-coady/trpo
* Value is trained on discounted reward sum
* Advantage is estimated with GAE
* Policy is not parametrized (i.e. no Neural Network for policies). 

### Classic Control: 
* Three algorithms implemented: DR TRPO KL, DR TRPO Wasserstein, A2C (baseline)
* This version can run on OpenAI Gym classic control environment. Parameters in the implemention are set up specifically for 'CartPole-v1' environment. 
* Value is trained on discounted reward sum
* Advantage is estimated with Monte Carlo Estimator
* Policy is parameterized as a neural net that maps from state s to the PMF of π(a|s).
