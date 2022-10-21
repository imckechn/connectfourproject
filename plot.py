import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

vr100 = pd.read_pickle('./data/2vr100.pkl')
vr200 = pd.read_pickle('./data/2vr200.pkl')
vr300 = pd.read_pickle('./data/2vr300.pkl')
vr400 = pd.read_pickle('./data/2vr400.pkl')
vr500 = pd.read_pickle('./data/2vr500.pkl')
vr600 = pd.read_pickle('./data/2vr600.pkl')
vr700 = pd.read_pickle('./data/2vr700.pkl')
vrp = pd.read_pickle('./data/2vrp.pkl')

f, ax = plt.subplots(1,1)
x_col = 'rollouts'
y_col = 'score'

sns.lineplot(x=vr100[x_col], y=vr100[y_col], label='Rollout (n=100)').set(title="score (mean of 300 games) vs. rollouts as second player")
sns.lineplot(x=vr200[x_col], y=vr200[y_col], label='Rollout (n=200)')
sns.lineplot(x=vr300[x_col], y=vr300[y_col], label='Rollout (n=300)')
sns.lineplot(x=vr400[x_col], y=vr400[y_col], label='Rollout (n=400)')
sns.lineplot(x=vr500[x_col], y=vr500[y_col], label='Rollout (n=500)')
sns.lineplot(x=vr600[x_col], y=vr600[y_col], label='Rollout (n=600)')
sns.lineplot(x=vr700[x_col], y=vr700[y_col], label='Rollout (n=700)')
sns.lineplot(x=vrp[x_col], y=vrp[y_col], label='Random Plus')
plt.axhline(linestyle='--', lw=1, color='black')
plt.legend(fontsize='x-small', title='opponent')
plt.savefig('./plots/ROandRPvsRO.png')