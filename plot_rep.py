import pickle
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

data1 = pickle.load(open('rep_1_data.pk', 'rb'))
data2 = pickle.load(open('rep_2_data.pk', 'rb'))
data3 = pickle.load(open('rep_3_data.pk', 'rb'))


print(data1.head())
print(data2.head())
print(data3.head())


fig, axes = plt.subplots(nrows=1, ncols=2, sharey=True)

sns.lineplot(data=data1, x='epoch', y='CEL', label='42 inputs to NN', ax=axes[0])
sns.lineplot(data=data2, x='epoch', y='CEL', linestyle='--', label='84 inputs to NN', ax=axes[0])
sns.lineplot(data=data3, x='epoch', y='CEL', linestyle='-.', label='126 inputs to NN', ax=axes[0])

data1 = data1.head(500)
data2 = data2.head(500)
data3 = data3.head(500)

sns.lineplot(data=data1, x='epoch', y='CEL', label='42 inputs to NN', ax=axes[1])
sns.lineplot(data=data2, x='epoch', y='CEL', linestyle='--', label='84 inputs to NN', ax=axes[1])
sns.lineplot(data=data3, x='epoch', y='CEL', linestyle='-.', label='126 inputs to NN', ax=axes[1])

fig.set_size_inches(8, 4)
fig.savefig('./plots/representation/compare-1v2v3-both.png')
