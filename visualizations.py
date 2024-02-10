import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

iris = sns.load_dataset('iris')

print(iris.loc[iris['species'] ==  'setosa'].value_counts())

sns.pairplot(data = iris, hue = 'species', palette = 'viridis', diag_kind = 'hist')
plt.savefig('./plots/pairplot.png')

sns.kdeplot(data = iris.loc[iris['species'] == 'setosa'], x = 'sepal_length', y = 'sepal_width', fill = True)
plt.savefig('./plots/kdeplot.png')
plt.show()

