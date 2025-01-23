r"""This code applies the moments accountant (MA), Dual and Central Limit 
Theorem (CLT) to estimate privacy budget of an iterated subsampled 
Gaussian Mechanism (either uniformly or by Poisson subsampling). 
The mechanism's parameters are controlled by flags.

Example:
  compute_muP
    --N=60000 \
    --batch_size=256 \
    --noise_multiplier=1.3 \
    --epochs=15

The output states that DP-optimizer satisfies 0.227-GDP.
"""

import numpy as np
from scipy.stats import norm
from scipy import optimize

# Total number of examples:N
# batch size:batch_size
# Noise multiplier for DP-SGD/DP-Adam:noise_multiplier
# current epoch:epoch
# Target delta:delta




'''
from gdp_accountant import compute_epsP, compute_epsilon

compute_epsilon: 给定sigma，求MA-DP的epsilon


###### GDP ######
# GDP和sigma转换关系
# 给定sigma，求GDP的mu  -->  可以转换成：给定mu，求sigma
def compute_mu_poisson(epoch, noise_multi, n, batch_size):
  """Compute mu from Poisson subsampling."""
  t = epoch * n / batch_size
  return np.sqrt(np.exp(noise_multi**(-2)) - 1) * np.sqrt(t) * batch_size / n

# MA和GDP转换关系
# 给定mu，求GDP的eps  -->  可以转换成：给定eps，求mu
def delta_eps_mu(eps, mu):
  """Compute dual between mu-GDP and (epsilon, delta)-DP."""
  return norm.cdf(-eps / mu +
                  mu / 2) - np.exp(eps) * norm.cdf(-eps / mu - mu / 2)
def eps_from_mu(mu, delta):
  """Compute epsilon from mu given delta via inverse dual."""
  def f(x):
    """Reversely solve dual by matching delta."""
    return delta_eps_mu(x, mu) - delta
  return optimize.root_scalar(f, bracket=[0, 500], method='brentq').root

'''


'''
n, batch_size, target_epsilon, epochs, delta, min_noise = 60000, 256, 1.0, 15, 1e-6, 1e-5

###### DP ######
def DP_eps_to_sigma(eps, delta):
    return np.sqrt(2 * np.log(1.25/delta)) / eps
sigma_DP = DP_eps_to_sigma(target_epsilon, delta)
print('sigma_DP:\t', sigma_DP)


###### RDP ######
from tensorflow_privacy.privacy.analysis.compute_noise_from_budget_lib import compute_noise as RDP_eps_to_sigma
sigma_RDP = RDP_eps_to_sigma(n, batch_size, target_epsilon, epochs, delta, min_noise)
print('sigma_RDP:\t', sigma_RDP)


###### GDP ######
def GDP_eps_to_mu(eps, delta):
    def transform_MA_and_GDP(eps, mu):
        return norm.cdf(-eps / mu + mu / 2) - np.exp(eps) * norm.cdf(-eps / mu - mu / 2)
    def f(x):
        # 求隐函数的解
        return transform_MA_and_GDP(eps, x) - delta
    return optimize.root_scalar(f, bracket=[0.0001, 100], method='toms748').root

def GDP_mu_to_sigma(epochs, mu, n, batch_size):
    def GDP_sigma_to_mu(epochs, sigma, n, batch_size):
        t = epochs * n / batch_size 
        return np.sqrt(np.exp(sigma**(-2)) - 1) * np.sqrt(t) * batch_size / n
    def f(x):
        # 求隐函数的解
        return GDP_sigma_to_mu(epochs, x, n, batch_size) - mu
    return optimize.root_scalar(f, bracket=[0.05, 100], method='toms748').root
mu = GDP_eps_to_mu(target_epsilon, delta)
# print('GDP_mu:\t', mu)
sigma_GDP = GDP_mu_to_sigma(epochs, mu, n, batch_size)
print('sigma_GDP:\t', sigma_GDP)




###### RDP ######


sys.exit()



from tensorflow_privacy.privacy.analysis import compute_noise_from_budget_lib

n, batch_size, target_epsilon, epochs,delta, min_noise = 100000000, 1024, 5907984.81339406, 10, 1e-7, 1e-5

target_noise = compute_noise_from_budget_lib.compute_noise(
        n, batch_size, target_epsilon, epochs, delta, min_noise)

print("target noise:\t", target_noise)
'''


from tensorflow_privacy.privacy.analysis.rdp_accountant import compute_rdp
from tensorflow_privacy.privacy.analysis.rdp_accountant import get_privacy_spent

def compute_epsilon(epoch,noise_multi,N,batch_size,delta):
  """Computes epsilon value for given hyperparameters."""
  orders = [1 + x / 10. for x in range(1, 100)] + list(np.arange(12, 60,0.2))+list(np.arange(60,100,1))
  sampling_probability = batch_size / N
  rdp = compute_rdp(q=sampling_probability,
                    noise_multiplier=noise_multi,
                    steps=epoch*N/batch_size,
                    orders=orders)
  return get_privacy_spent(orders, rdp, target_delta=delta)[0]

#print("eps:\t", compute_epsilon(epochs, target_noise, n, batch_size, delta))
#sys.exit()
'''
('Test0', 60000, 150, 0.941870567, 15, 1e-5, 1e-5, 1.3)
('Test1', 100000, 100, 1.70928734, 30, 1e-7, 1e-6, 1.0)
('Test2', 100000000, 1024, 5907984.81339406, 10, 1e-7, 1e-5, 0.1)
('Test3', 100000000, 1024, 5907984.81339406, 10, 1e-7, 1, 0)

compute_noise(self, n, batch_size, target_epsilon, epochs,delta, min_noise, expected_noise):
'''



from sympy.solvers import solve
from sympy import Symbol



# Compute mu from uniform subsampling
def compute_muU(epoch,noise_multi,N,batch_size):
    T=epoch*N/batch_size
    c=batch_size*np.sqrt(T)/N
    return(np.sqrt(2)*c*np.sqrt(np.exp(noise_multi**(-2))*norm.cdf(1.5/noise_multi)+3*norm.cdf(-0.5/noise_multi)-2))

# Compute mu from Poisson subsampling
def compute_muP(epoch,noise_multi,N,batch_size):
    T=epoch*N/batch_size
    return(np.sqrt(np.exp(noise_multi**(-2))-1)*np.sqrt(T)*batch_size/N)
    
# Dual between mu-GDP and (epsilon,delta)-DP
def delta_eps_mu(eps,mu):
    return norm.cdf(-eps/mu+mu/2)-np.exp(eps)*norm.cdf(-eps/mu-mu/2)

# inverse Dual
def eps_from_mu(mu,delta):
    def f(x):
        return delta_eps_mu(x,mu)-delta    
    return optimize.root_scalar(f, bracket=[0, 500], method='brentq').root

# inverse Dual of uniform subsampling
def compute_epsU(epoch,noise_multi,N,batch_size,delta):
    return(eps_from_mu(compute_muU(epoch,noise_multi,N,batch_size),delta))

# inverse Dual of Poisson subsampling
def compute_epsP(epoch,noise_multi,N,batch_size,delta):
    return(eps_from_mu(compute_muP(epoch,noise_multi,N,batch_size),delta))

from tensorflow_privacy.privacy.analysis.rdp_accountant import compute_rdp
from tensorflow_privacy.privacy.analysis.rdp_accountant import get_privacy_spent

# Compute epsilon by MA
def compute_epsilon(epoch,noise_multi,N,batch_size,delta):
  """Computes epsilon value for given hyperparameters."""
  orders = [1 + x / 10. for x in range(1, 100)] + list(np.arange(12, 60,0.2))+list(np.arange(60,100,1))
  sampling_probability = batch_size / N
  rdp = compute_rdp(q=sampling_probability,
                    noise_multiplier=noise_multi,
                    steps=epoch*N/batch_size,
                    orders=orders)
  return get_privacy_spent(orders, rdp, target_delta=delta)[0]




























