""" This script contains code to create the training, pool and test datasets. """ 

# importing the libraries
import pandas as pd
import numpy as np 
import os 
from sklearn import preprocessing 
from numpy.random import RandomState

# define variables
dataset_name = "sf2"
train_size = 0.1
pool_size = 0.7
test_size = 0.2
data_path = "../2.Datasets/{}.csv".format(dataset_name)

# read the pandas dataframe and obtain the amount of targets 
df = pd.read_csv(data_path)

col_names = list(df.columns)
target_fts_length = 0

for name in col_names: 
    if 'target' in name:
        target_fts_length += 1

# normalize the target values
for i in range(target_fts_length):
    col_name = col_names[-(i+1)]
    targets = df[col_name].tolist()
    targets_norm = np.round(preprocessing.normalize(np.array(targets).reshape(1, -1))[0].tolist(), 5)
    dcol = pd.DataFrame({col_name: targets_norm})
    df[col_name] = dcol[col_name]

# divide the datasets
rng = RandomState()
for i in range(5):
    rest = df.sample(frac=(train_size+pool_size), random_state=rng)
    restcop = rest.copy()
    test = df.loc[~df.index.isin(rest.index)]

    train = rest.sample(frac=(train_size/(train_size+pool_size)), random_state=rng)
    pool = rest = rest.loc[~rest.index.isin(train.index)]

    # store as csv files
    folder_path = "../2.Datasets/{}".format(dataset_name)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    datasets = {'train': train, 'pool': pool, 'test': test, 'train+pool': restcop}

    for dataset in datasets:
        datasets[dataset].to_csv("../2.Datasets/{}/{}_{}.csv".format(dataset_name, dataset, i+1), index=False)