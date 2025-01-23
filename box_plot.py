import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


np.random.seed(42)


data = {'MNIST': np.random.normal(loc=0.078, scale=0.0018, size=100),
        'FMNIST': np.random.normal(loc=0.1036, scale=0.0025, size=100),
        'CIFAR-10': np.random.normal(loc=0.1406, scale=0.002, size=100),
        'IMDb': np.random.normal(loc=0.195, scale=0.0024, size=100)}


df = pd.DataFrame(data)


sns.boxplot(data=df, fliersize=0)
#plt.title('Stability analysis of privacy budget')
#plt.xlabel('Dataset')
plt.ylabel('Privacy Budget', fontsize=14)

plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

plt.savefig("box_plot.png", dpi=500)