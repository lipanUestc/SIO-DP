import numpy as np
import os
from privacy_accountant import RDP_eps_to_sigma, GDP_eps_to_mu, GDP_mu_to_sigma, SIO_eps_to_mu, SIO_mu_to_sigma


os.system("python estimate_complexity.py --dataset {} --reference_model_num {}".format(dataset, reference_model_num))
os.system("python bootstrap_resampling.py --dataset {}".format(dataset))







sys.exit()




for dataset in ['fmnist']: #'cifar10'/'mnist'/'fmnist'/'imdb'
	
	### Parameters ###
	if dataset=='cifar10':
		n, batch_size, epochs, delta, min_noise, reference_model_num = 50000, 256, 100, 1e-5, 1e-5, 5
	elif dataset=='imdb':
		n, batch_size, epochs, delta, min_noise, reference_model_num = 25000, 256, 10, 1e-5, 1e-5, 5
	elif dataset=='mnist':
		n, batch_size, epochs, delta, min_noise, reference_model_num = 60000, 256, 10, 1e-5, 1e-5, 5
	elif dataset=='fmnist':
		n, batch_size, epochs, delta, min_noise, reference_model_num = 60000, 256, 10, 1e-5, 1e-5, 5

	'''
	### Train Reference Models ###
	for i in range(1, reference_model_num+1):
		f = open("experiment_results.txt", 'a')
		f.write("{} Reference Model {}\n".format(dataset, i))
		f.write("Dataset_size:{} Batch_size:{} Epochs:{}\n".format(n, batch_size, epochs))
		f.close()es
		os.system("python {}.py --disable-dp --checkpoint-file {}_reference_model_{} --seed {}".format(dataset, dataset, i, i))

	sys.exit()
	'''
	### Analysis Complexity ###
	# estimate complexity of each reference model
	# os.system("python estimate_complexity.py --dataset {} --reference_model_num {}".format(dataset, reference_model_num))
	
	# Use bootstrap to derive the upper bound of multiple complexity estimates 
	# os.system("python bootstrap_resampling.py --dataset {}".format(dataset))
	

	### Train DP Model ###
	for target_epsilon in [1.0, 4.0]: #[0.5, 1.0, 2.0, 3.0, 4.0]
		sigma_RDP = RDP_eps_to_sigma(n, batch_size, target_epsilon, epochs, delta, min_noise)

		mu = GDP_eps_to_mu(target_epsilon, delta)
		sigma_GDP = GDP_mu_to_sigma(epochs, mu, n, batch_size)

		complexity = np.load("{}_complexity_final.npy".format(dataset))
		mu = SIO_eps_to_mu(target_epsilon, delta)
		sigma_SIO = SIO_mu_to_sigma(epochs, mu, n, batch_size, complexity)

		f = open("experiment_results.txt", 'a')
		f.write("{} RDP --> eps:{} \t sigma:{}\n".format(dataset, target_epsilon, sigma_RDP))
		f.write("Dataset_size:{} Batch_size:{} Epochs:{}\n".format(n, batch_size, epochs))
		f.close()
		os.system("python {}.py --sigma {} --checkpoint-file {}_RDP_eps_{}_model --batch-size {} --epochs {} --seed 0".format(dataset, sigma_RDP, dataset, target_epsilon, batch_size, epochs))
		
		f = open("experiment_results.txt", 'a')
		f.write("{} GDP --> eps:{} \t sigma:{}\n".format(dataset, target_epsilon, sigma_GDP))
		f.write("Dataset_size:{} Batch_size:{} Epochs:{}\n".format(n, batch_size, epochs))
		f.close()
		os.system("python {}.py --sigma {} --checkpoint-file {}_GDP_eps_{}_model --batch-size {} --epochs {} --seed 0".format(dataset, sigma_GDP, dataset, target_epsilon, batch_size, epochs))
		
		f = open("experiment_results.txt", 'a')
		f.write("{} SIO --> eps:{} \t sigma:{}\n".format(dataset, target_epsilon, sigma_SIO))
		f.write("Dataset_size:{} Batch_size:{} Epochs:{}\n".format(n, batch_size, epochs))
		f.close()
		os.system("python {}.py --sigma {} --checkpoint-file {}_SIO_eps_{}_model --batch-size {} --epochs {} --seed 0".format(dataset, sigma_SIO, dataset, target_epsilon, batch_size, epochs))












sys.exit()









### Parameters ###
dataset = "cifar10"
reference_model_num = 5


'''
### Train Reference Models ###
	--disable-dp --> train reference models
	--checkpoint-file --> save name of checkpoint
	--seed --> set random seed
'''
for i in range(1, reference_model_num+1):
	os.system("python {}.py --disable-dp --checkpoint-file {}_reference_model_{} --seed {}".format(dataset, dataset, i, i))


'''
### Estimate the Complexity Matrix ###
	complexity matrix.shape = len(train_dataset) * reference_model_num
'''
os.system("python estimate_complexity.py --dataset {} --reference_model_num {}".format(dataset, reference_model_num))
os.system("python bootstrap_resampling.py")







'''
### Given a required epsilon, calculates the corresponding sigma (noise scale) ###
	n --> total number of training dataset
	batch_size --> size of mini-batch
	target_epsilon --> use epsilon to indicate our required privacy protection
	epochs --> total training iterations
	delta --> another privacy parameter, serving as a relax fator
	min_noise --> minimal scale of noise addition 
'''
n, batch_size, target_epsilon, epochs, delta, min_noise = 50000, 256, 1.0, 100, 1e-5, 1e-5

from privacy_accountant import SIO_eps_to_mu, SIO_mu_to_sigma
mu = SIO_eps_to_mu(target_epsilon, delta)
sigma_SIO = SIO_mu_to_sigma(epochs, mu, n, batch_size)





'''
### Train DP model ###
'''
os.system("python cifar10.py --sigma {} --checkpoint-file cifar10_eps_{}_model --batch-size 256 --epochs 100 --seed 0".format(sigma_SIO, target_epsilon))













