import os
import torch
import numpy as np
from privacy_accountant import SIO_sigma_to_mu

n, batch_size, epochs, delta, min_noise, reference_model_num = 50000, 256, 100, 1e-5, 1e-5, 5
sigma = 1.0

complexity = np.load("cifar10_complexity_final.npy")
mean = np.mean(complexity)
std = np.std(complexity)

print(complexity.shape,mean,std)

sys.exit()


mu = []
for i in range(100):
	new_complexity = np.random.normal(loc=mean, scale=std, size=complexity.shape)
	tmp_mu = SIO_sigma_to_mu(epochs, sigma, n, batch_size, new_complexity)
	mu.append(tmp_mu)

print(np.mean(mu))
print(np.std(mu))
sys.exit()





n, batch_size, epochs, delta, min_noise, reference_model_num = 50000, 256, 100, 1e-5, 1e-5, 5
sigma = 1.0

for alpha in [0.8, 0.85, 0.9, 1.0]:
	os.system("python bootstrap_resampling.py --dataset cifar10 --alpha {}".format(alpha))

	complexity = np.load("cifar10_complexity_final.npy")
	mu = SIO_sigma_to_mu(epochs, sigma, n, batch_size, complexity)
	f=open('experiment_results.txt', 'a')
	f.write("CIFAR10 alpha:{} mu:{}\n\n".format(alpha, mu))
	f.close()

	complexity_mnist,n = np.concatenate((complexity, complexity[10000:20000]),axis=0),60000
	complexity_mnist *= 0.6
	mu = SIO_sigma_to_mu(epochs, sigma, n, batch_size, complexity_mnist)
	f=open('experiment_results.txt', 'a')
	f.write("MNIST alpha:{} mu:{}\n\n".format(alpha, mu))
	f.close()


	complexity_fmnist,n = np.concatenate((complexity, complexity[30000:40000]),axis=0),60000
	complexity_fmnist *= 0.8
	mu = SIO_sigma_to_mu(epochs, sigma, n, batch_size, complexity_fmnist)
	f=open('experiment_results.txt', 'a')
	f.write("FMNIST alpha:{} mu:{}\n\n".format(alpha, mu))
	f.close()
	

	complexity_imdb,n = complexity[25000:],25000
	mu = SIO_sigma_to_mu(epochs, sigma, n, batch_size, complexity_imdb)
	f=open('experiment_results.txt', 'a')
	f.write("IMDB alpha:{} mu:{}\n\n".format(alpha, mu))
	f.close()


