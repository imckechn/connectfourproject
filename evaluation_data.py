import pandas as pd
import pickle

eval = pickle.load(open('./datasets/ucb_net_v9/dataset_19_evaluation_data.pk', 'rb'))
eval.insert(0, 'agent0_generation', eval['generation'] - 1)
eval.insert(1, 'agent1_generation', eval['generation'])

eval['agent0_generation'] = eval['generation'] - 1
eval['agent1_generation'] = eval['generation']

eval = eval.drop(labels='generation', axis=1)

eval.to_csv('./datasets/evaluation.csv')
