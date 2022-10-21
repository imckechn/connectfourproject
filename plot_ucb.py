import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_pickle('./data/ucb/ro0')

f, ax = plt.subplots(1,1)
x_col = 'confidence_levels'
y_col = 'score'
sem_col = 'sem'

data = pd.read_pickle('./data/ucb/ro0')
sns.lineplot(data=data, x=x_col, y=y_col, label='Rollout (n=100)').set(title="score (mean of 500 games) on confidence level as first player")
plt.fill_between(data[x_col], data[y_col] - data[sem_col]/2, data[y_col] + data[sem_col]/2, color='tab:blue', alpha=0.2)

data = pd.read_pickle('./data/ucb/ro1')
sns.lineplot(data=data, x=x_col, y=y_col, label='Rollout (n=200)')
plt.fill_between(data[x_col], data[y_col] - data[sem_col]/2, data[y_col] + data[sem_col]/2, color='tab:blue', alpha=0.2)

data = pd.read_pickle('./data/ucb/ro2')
sns.lineplot(data=data, x=x_col, y=y_col, label='Rollout (n=300)')
plt.fill_between(data[x_col], data[y_col] - data[sem_col]/2, data[y_col] + data[sem_col]/2, color='tab:blue', alpha=0.2)

data = pd.read_pickle('./data/ucb/ro3')
sns.lineplot(data=data, x=x_col, y=y_col, label='Rollout (n=400)')
plt.fill_between(data[x_col], data[y_col] - data[sem_col]/2, data[y_col] + data[sem_col]/2, color='tab:blue', alpha=0.2)

data = pd.read_pickle('./data/ucb/ro4')
sns.lineplot(data=data, x=x_col, y=y_col, label='Rollout (n=500)')
plt.fill_between(data[x_col], data[y_col] - data[sem_col]/2, data[y_col] + data[sem_col]/2, color='tab:blue', alpha=0.2)

data = pd.read_pickle('./data/ucb/ro5')
sns.lineplot(data=data, x=x_col, y=y_col, label='Rollout (n=600)')
plt.fill_between(data[x_col], data[y_col] - data[sem_col]/2, data[y_col] + data[sem_col]/2, color='tab:blue', alpha=0.2)

data = pd.read_pickle('./data/ucb/ro6')
sns.lineplot(data=data, x=x_col, y=y_col, label='Rollout (n=700)')
plt.fill_between(data[x_col], data[y_col] - data[sem_col]/2, data[y_col] + data[sem_col]/2, color='tab:blue', alpha=0.2)

plt.axhline(linestyle='--', lw=1, color='black')
plt.legend(fontsize='x-small', title='opponent')
plt.savefig('./plots/ucb-rollout/ucbcl1.png')
plt.clf()

data = pd.read_pickle('./data/ucb/2ro0')
sns.lineplot(data=data, x=x_col, y=y_col, label='Rollout (n=100)').set(title="score (mean of 500 games) on confidence level as second player")
plt.fill_between(data[x_col], data[y_col] - data[sem_col]/2, data[y_col] + data[sem_col]/2, color='tab:blue', alpha=0.2)

data = pd.read_pickle('./data/ucb/2ro1')
sns.lineplot(data=data, x=x_col, y=y_col, label='Rollout (n=200)')
plt.fill_between(data[x_col], data[y_col] - data[sem_col]/2, data[y_col] + data[sem_col]/2, color='tab:blue', alpha=0.2)

data = pd.read_pickle('./data/ucb/2ro2')
sns.lineplot(data=data, x=x_col, y=y_col, label='Rollout (n=300)')
plt.fill_between(data[x_col], data[y_col] - data[sem_col]/2, data[y_col] + data[sem_col]/2, color='tab:blue', alpha=0.2)

data = pd.read_pickle('./data/ucb/2ro3')
sns.lineplot(data=data, x=x_col, y=y_col, label='Rollout (n=400)')
plt.fill_between(data[x_col], data[y_col] - data[sem_col]/2, data[y_col] + data[sem_col]/2, color='tab:blue', alpha=0.2)

data = pd.read_pickle('./data/ucb/2ro4')
sns.lineplot(data=data, x=x_col, y=y_col, label='Rollout (n=500)')
plt.fill_between(data[x_col], data[y_col] - data[sem_col]/2, data[y_col] + data[sem_col]/2, color='tab:blue', alpha=0.2)

data = pd.read_pickle('./data/ucb/2ro5')
sns.lineplot(data=data, x=x_col, y=y_col, label='Rollout (n=600)')
plt.fill_between(data[x_col], data[y_col] - data[sem_col]/2, data[y_col] + data[sem_col]/2, color='tab:blue', alpha=0.2)

data = pd.read_pickle('./data/ucb/2ro6')
sns.lineplot(data=data, x=x_col, y=y_col, label='Rollout (n=700)')
plt.fill_between(data[x_col], data[y_col] - data[sem_col]/2, data[y_col] + data[sem_col]/2, color='tab:blue', alpha=0.2)

plt.axhline(linestyle='--', lw=1, color='black')
plt.legend(fontsize='x-small', title='opponent')
plt.savefig('./plots/ucb-rollout/ucbcl2.png')



