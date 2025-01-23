import numpy as np
from scipy.stats import norm
from scipy import optimize
import tensorflow_privacy

'''
n --> total number of training dataset
batch_size --> size of mini-batch
target_epsilon --> use epsilon to indicate our required privacy protection
epochs --> total training iterations
delta --> another privacy parameter, serving as a relax fator
min_noise --> minimal scale of noise addition 
'''
# n, batch_size, target_epsilon, epochs, delta, min_noise = 60000, 256, 1.0, 15, 1e-5, 1e-5


###### DP ######
def DP_eps_to_sigma(eps, delta):
    return np.sqrt(2 * np.log(1.25/delta)) / eps
# sigma_DP = DP_eps_to_sigma(target_epsilon, delta)
# print('sigma_DP:\t', sigma_DP)


###### RDP ######
from tensorflow_privacy.privacy.analysis.compute_noise_from_budget_lib import compute_noise as RDP_eps_to_sigma
# sigma_RDP = RDP_eps_to_sigma(n, batch_size, target_epsilon, epochs, delta, min_noise)
# print('sigma_RDP:\t', sigma_RDP)


###### GDP ######
def GDP_eps_to_mu(eps, delta):
    def transform_MA_and_GDP(eps, mu):
        return norm.cdf(-eps / mu + mu / 2) - np.exp(eps) * norm.cdf(-eps / mu - mu / 2)
    def f(x):
        return transform_MA_and_GDP(eps, x) - delta
    return optimize.root_scalar(f, bracket=[0.0001, 100], method='toms748').root

def GDP_mu_to_sigma(epochs, mu, n, batch_size):
    def GDP_sigma_to_mu(epochs, sigma, n, batch_size):
        t = epochs * n / batch_size  # total number of DP composition
        return np.sqrt(np.exp(sigma**(-2)) - 1) * np.sqrt(t) * batch_size / n
    def f(x):
        return GDP_sigma_to_mu(epochs, x, n, batch_size) - mu
    return optimize.root_scalar(f, bracket=[0.2, 100], method='toms748').root
#mu = GDP_eps_to_mu(target_epsilon, delta)
#sigma_GDP = GDP_mu_to_sigma(epochs, mu, n, batch_size)
#print('sigma_GDP:\t', sigma_GDP)



###### SIO-DP ######
#complexity = np.load("cifar10_complexity_final.npy")
def SIO_eps_to_mu(eps, delta):
    def transform_MA_and_GDP(eps, mu):
        return norm.cdf(-eps / mu + mu / 2) - np.exp(eps) * norm.cdf(-eps / mu - mu / 2)
    def f(x):
        return transform_MA_and_GDP(eps, x) - delta
    return optimize.root_scalar(f, bracket=[0.0001, 100], method='toms748').root

def SIO_sigma_to_mu(epochs, sigma, n, batch_size, complexity):
    t = epochs * n / batch_size 
    mu_i_square = np.square(complexity)/(sigma**2)
    return np.sqrt(np.mean(np.exp(mu_i_square)) - 1) * np.sqrt(t) * batch_size / n

def SIO_mu_to_sigma(epochs, mu, n, batch_size, complexity):
    def f(x):
        return SIO_sigma_to_mu(epochs, x, n, batch_size, complexity) - mu
    return optimize.root_scalar(f, bracket=[0.2, 100], method='toms748').root
#mu = SIO_eps_to_mu(target_epsilon, delta)
#sigma_SIO = SIO_mu_to_sigma(epochs, mu, n, batch_size, complexity)
#print('sigma_SIO:\t', sigma_SIO)























