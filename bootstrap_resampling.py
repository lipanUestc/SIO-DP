import argparse
import numpy as np
import random as rd
from tqdm import tqdm


parser = argparse.ArgumentParser(description="PyTorch CIFAR10 DP Training")
parser.add_argument("--dataset",default="cifar10",type=str,help="dataset",)
parser.add_argument("--bootstrap_replicates_num", default=10000, type=int, help="we create X bootstrap replicates for each sample")
parser.add_argument("--reference_model_num", default=5, type=int, help="the number of reference models. ")
parser.add_argument("--alpha",default=0.95,type=float,help="confidence_level",)
args = parser.parse_args()

def generate_bootstrap_replicates(data, nboot, replacement=True):
    """
        Generate n=nboot bootstrap replicates of the data with/without replacement,
        and return a 2D numpy array of them.

        Input:    data  (anything numpy can handle as a numpy array), 
                  nboot (the number of generated bootstrap replicates)
        Output:   2D numpy array of size (nboot x length of data)
    """

    # Ensure that our data is a 1D array.
    data = np.ravel(data)

    # Create a 2D array of bootstrap samples indexes
    if replacement==True: # with replacement (note: 50x-ish faster than without)
        idx = np.random.randint(data.size, size=(nboot, data.size))
    elif replacement==False: # without replacement
        idx = np.vstack([np.random.permutation(data.size) for x in np.arange(nboot)])

    return data[idx]


###################################################
def confidence_interval(data, stat=np.mean, nboot=10000, replacement=True, alpha=0.05, method='pi', keepboot=False):
    """
        Compute the (1-alpha) confidence interval of a statistic (i.e.: mean, median, etc)
        of the data using bootstrap resampling.
        
        Arguments:
            stat:        statistics we want the confidence interval for (must be a function)
            nboot:       number of bootstrap samples to generate
            replacement: resampling done with (True) or without (False) replacement
            alpha:       level of confidence interval
            method:      type of bootstrap we want to perform
            keepboot:    if True, return the nboot bootstrap statistics from which
                         the confidence intervals are extracted
        
        Methods available:
            - 'pi' = Percentile Interval
            - 'bca' = Bias-Corrected Accelerated Interval (available soon)
    """

    # apply bootstrap to data
    boot = generate_bootstrap_replicates(data, nboot=nboot, replacement=replacement)

    # calculate the statistics for each bootstrap sample and sort them
    sorted_stat = np.sort(stat(boot, axis=1))

    # Percentile Interval method (for the moment the only one available) 
    if method == 'pi':
        beta = 1-alpha
        ci = (sorted_stat[np.round(nboot*beta/2).astype(int)], 
              sorted_stat[np.round(nboot*(1-beta/2)).astype(int)])

    if keepboot == True:
        return ci, sorted_stat
    else:
        return ci


final_complexity = []
complexity_matrix = np.load("{}_complexity_matrix.npy".format(args.dataset))[:,:args.reference_model_num]

print("Starting bootstrap analysis...")
for i in tqdm(range(complexity_matrix.shape[0])):
    data = complexity_matrix[i]
    (lower_bound, upper_bound) = confidence_interval(data, nboot=args.bootstrap_replicates_num, alpha=args.alpha)
    final_complexity.append(upper_bound)

np.save("{}_complexity_final.npy".format(args.dataset), np.array(final_complexity))








