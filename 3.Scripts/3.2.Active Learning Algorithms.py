""" This script contains code of the active learning algorithms for multi-target regression. """ 

# importing the libraries
import csv
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, pairwise_distances_argmin_min, auc
from sklearn.cluster import KMeans
from sklearn import preprocessing 


""" The class for the active learning algorithm. """
from ActiveLearning import *

""" The subclass for QBC-RF. """
from QBCRF import *

""" The subclass for the upper bound method. """
from UpperBound import *

""" The subclass for random sampling. """
from RandomSampling import *

""" The subclass for the baseline method. """
from GreedySampling import *

""" The training loop for all the methods. """

dataset = "edm"
Method = activelearning(dataset)
# read the first version of the datasets to define the batch size
X_train, y_train, _, _ = Method.data_read('train_{}'.format(1))
X_pool, y_pool, n_pool, target_length = Method.data_read('pool_{}'.format(1))
X_rest, y_rest, _, _ = Method.data_read('train+pool_{}'.format(1))
X_test, y_test, _, _ = Method.data_read('test_{}'.format(1))

batch_percentage = 5
batch_size = round((batch_percentage/100) * len(X_pool)) 
n_epochs = 15

# define the different methods
proposed_method_instance = instancebased(batch_size, n_epochs)
upperbound_method = upperbound(n_epochs)
lowerbound_method = lowerbound(batch_size, n_epochs)
baseline_method = baseline(batch_size, n_epochs)

# make the folders to store the results
folder_path = "../4.Results/{}".format(dataset)
if not os.path.exists(folder_path):
    os.makedirs("../4.Results/{}/R2".format(dataset))
    os.makedirs("../4.Results/{}/MSE".format(dataset))
    os.makedirs("../4.Results/{}/MAE".format(dataset))
    os.makedirs("../4.Results/{}/PREDS".format(dataset))

for i in range(Method.iterations):
    if i > 0:
        X_train, y_train, _, _ = Method.data_read('train_{}'.format(i+1))
        X_pool, y_pool, n_pool, target_length = Method.data_read('pool_{}'.format(i+1))
        X_rest, y_rest, _, _ = Method.data_read('train+pool_{}'.format(i+1))
        X_test, y_test, _, _ = Method.data_read('test_{}'.format(i+1))

    # Copy the original datasets for the instance based, random and greedy method
    X_train_instance, X_pool_instance, y_train_instance, y_pool_instance = X_train.copy(), X_pool.copy(), y_train.copy(), y_pool.copy()
    X_train_random, X_pool_random, y_train_random, y_pool_random = X_train.copy(), X_pool.copy(), y_train.copy(), y_pool.copy()
    X_train_baseline, X_pool_baseline, y_train_baseline, y_pool_baseline = X_train.copy(), X_pool.copy(), y_train.copy(), y_pool.copy()
    
    print("\n" + 50*"-" + "Iteration {}".format(i+1) + 50*"-" + "\n")
    # QBC-RF    
    print("\n" + 50*"-" + "Instance based method" + 50*"-" + "\n")
    cols = ["Target_{}".format(i+1) for epoch in range(n_epochs) for i in range(target_length)]
    R2, MSE, MAE, Y_pred_df, instances_pool_qbc, targets_pool_qbc = proposed_method_instance.training(X_train_instance, X_pool_instance, X_test, y_train_instance, y_pool_instance, y_test, target_length)
    proposed_method_instance.instance_R2[i,:] = R2
    proposed_method_instance.instance_MSE[i,:] = MSE
    proposed_method_instance.instance_MAE[i,:] = MAE
    Y_pred_df.to_csv("../4.Results/{}/PREDS/Instance_preds_{}.csv".format(dataset, i), header=cols)
    instances_pool_qbc_df = pd.DataFrame(instances_pool_qbc)
    targets_pool_qbc_df = pd.DataFrame(targets_pool_qbc)
    instances_pool_qbc_df.to_csv("../4.Results/{}/Transfer_instances_QBC_{}.csv".format(dataset, i))
    targets_pool_qbc_df.to_csv("../4.Results/{}/Transfer_targets_QBC_{}.csv".format(dataset, i))

    # Upperbound 
    print("\n" + 50*"-" + "Upperbound method" + 50*"-" + "\n")
    cols = ["Target_{}".format(i+1) for i in range(target_length)]
    R2, MSE, MAE, Y_pred_df = upperbound_method.training(X_rest, X_test, y_rest, y_test, target_length)
    upperbound_method.upperbound_R2[i,:] = R2
    upperbound_method.upperbound_MSE[i,:] = MSE
    upperbound_method.upperbound_MAE[i,:] = MAE
    Y_pred_df.to_csv("../4.Results/{}/PREDS/Upperbound_preds_{}.csv".format(dataset, i), header=cols)

    # Random sampling
    print("\n" + 50*"-" + "Lowerbound method" + 50*"-" + "\n")
    cols = ["Target_{}".format(i+1) for epoch in range(n_epochs) for i in range(target_length)]
    R2, MSE, MAE, Y_pred_df = lowerbound_method.training(X_train_random, X_pool_random, X_test, y_train_random, y_pool_random, y_test, target_length)
    lowerbound_method.random_R2[i,:] = R2
    lowerbound_method.random_MSE[i,:] = MSE
    lowerbound_method.random_MAE[i,:] = MAE
    Y_pred_df.to_csv("../4.Results/{}/PREDS/Random_preds_{}.csv".format(dataset, i), header=cols)

    # Greedy sampling
    print("\n" + 50*"-" + "Baseline method" + 50*"-" + "\n")
    R2, MSE, MAE, Y_pred_df, instances_pool_baseline, targets_pool_baseline = baseline_method.training(X_train_baseline, X_pool_baseline, X_test, y_train_baseline, y_pool_baseline, y_test, target_length)
    baseline_method.baseline_R2[i,:] = R2
    baseline_method.baseline_MSE[i,:] = MSE
    baseline_method.baseline_MAE[i,:] = MAE
    Y_pred_df.to_csv("../4.Results/{}/PREDS/Greedy_preds_{}.csv".format(dataset, i), header=cols)
    instances_pool_baseline_df = pd.DataFrame(instances_pool_baseline)
    targets_pool_baseline_df = pd.DataFrame(targets_pool_baseline)
    instances_pool_baseline_df.to_csv("../4.Results/{}/Transfer_instances_Greedy_{}.csv".format(dataset, i))
    targets_pool_baseline_df.to_csv("../4.Results/{}/Transfer_targets_Greedy_{}.csv".format(dataset, i))

    # Plot the results

    plt.plot(proposed_method_instance.epochs, proposed_method_instance.instance_R2[i,:-1],'b', label='Instance based QBC')
    plt.plot(proposed_method_instance.epochs, upperbound_method.upperbound_R2[i,:-1],'r', label='Upper bound')
    plt.plot(proposed_method_instance.epochs, lowerbound_method.random_R2[i,:-1],'g', label='Random sampling')
    plt.plot(proposed_method_instance.epochs, baseline_method.baseline_R2[i,:-1],'y', label='Greedy sampling')
    plt.legend(loc='best')
    plt.xlabel('epoch')
    plt.ylabel('R2 score')
    plt.title('Instance based QBC method R2 performance for {} dataset: iteration {}'.format(Method.dataset_name, i+1))
    plt.savefig("../4.Results/{}/R2/R2_score_{}".format(dataset, i+1))
    plt.show()

    plt.plot(proposed_method_instance.epochs, proposed_method_instance.instance_MSE[i,:-1],'b', label='Instance based QBC')
    plt.plot(proposed_method_instance.epochs, upperbound_method.upperbound_MSE[i,:-1],'r', label='Upper bound')
    plt.plot(proposed_method_instance.epochs, lowerbound_method.random_MSE[i,:-1],'g', label='Random sampling')
    plt.plot(proposed_method_instance.epochs, baseline_method.baseline_MSE[i,:-1],'y', label='Greedy sampling')
    plt.legend(loc='best')
    plt.xlabel('epoch')
    plt.ylabel('MSE')
    plt.title('Instance based QBC method MSE performance for {} dataset: iteration {}'.format(Method.dataset_name, i+1))
    plt.savefig("../4.Results/{}/MSE/MSE_{}".format(dataset, i+1))
    plt.show()

    plt.plot(proposed_method_instance.epochs, proposed_method_instance.instance_MAE[i,:-1],'b', label='Instance based QBC')
    plt.plot(proposed_method_instance.epochs, upperbound_method.upperbound_MAE[i,:-1],'r', label='Upper bound')
    plt.plot(proposed_method_instance.epochs, lowerbound_method.random_MAE[i,:-1],'g', label='Random sampling')
    plt.plot(proposed_method_instance.epochs, baseline_method.baseline_MAE[i,:-1],'y', label='Greedy sampling')
    plt.legend(loc='best')
    plt.xlabel('epoch')
    plt.ylabel('MAE')
    plt.title('Instance based QBC method MAE performance for {} dataset: iteration {}'.format(Method.dataset_name, i+1))
    plt.savefig("../4.Results/{}/MAE/MAE_{}".format(dataset, i+1))
    plt.show()

instance_mean_R2, instance_mean_MSE, instance_mean_MAE = np.mean(proposed_method_instance.instance_R2, axis=0), np.mean(proposed_method_instance.instance_MSE, axis=0), np.mean(proposed_method_instance.instance_MAE, axis=0)
upperbound_mean_R2, upperbound_mean_MSE, upperbound_mean_MAE = np.mean(upperbound_method.upperbound_R2, axis=0), np.mean(upperbound_method.upperbound_MSE, axis=0), np.mean(upperbound_method.upperbound_MAE, axis=0)
random_mean_R2, random_mean_MSE, random_mean_MAE = np.mean(lowerbound_method.random_R2, axis=0), np.mean(lowerbound_method.random_MSE, axis=0), np.mean(lowerbound_method.random_MAE, axis=0)
greedy_mean_R2, greedy_mean_MSE, greedy_mean_MAE = np.mean(baseline_method.baseline_R2, axis=0), np.mean(baseline_method.baseline_MSE, axis=0), np.mean(baseline_method.baseline_MAE, axis=0)

plt.plot(proposed_method_instance.epochs, instance_mean_R2[:-1],'b', label='Instance based QBC')
plt.plot(proposed_method_instance.epochs, upperbound_mean_R2[:-1],'r', label='Upper bound')
plt.plot(proposed_method_instance.epochs, random_mean_R2[:-1],'g', label='Random sampling')
plt.plot(proposed_method_instance.epochs, greedy_mean_R2[:-1],'y', label='Greedy sampling')
plt.legend(loc='best')
plt.xlabel('epoch')
plt.ylabel('R2 score')
plt.title('Instance based QBC method R2 performance for {} dataset'.format(Method.dataset_name))
plt.savefig("../4.Results/{}/R2/R2".format(dataset))
plt.show()

plt.plot(proposed_method_instance.epochs, instance_mean_MSE[:-1],'b', label='Instance based QBC')
plt.plot(proposed_method_instance.epochs, upperbound_mean_MSE[:-1],'r', label='Upper bound')
plt.plot(proposed_method_instance.epochs, random_mean_MSE[:-1],'g', label='Random sampling')
plt.plot(proposed_method_instance.epochs, greedy_mean_MSE[:-1],'y', label='Greedy sampling')
plt.legend(loc='best')
plt.xlabel('epoch')
plt.ylabel('MSE')
plt.title('Instance based QBC method MSE performance for {} dataset'.format(Method.dataset_name))
plt.savefig("../4.Results/{}/MSE/MSE".format(dataset))
plt.show()

plt.plot(proposed_method_instance.epochs, instance_mean_MAE[:-1],'b', label='Instance based QBC')
plt.plot(proposed_method_instance.epochs, upperbound_mean_MAE[:-1],'r', label='Upper bound')
plt.plot(proposed_method_instance.epochs, random_mean_MAE[:-1],'g', label='Random sampling')
plt.plot(proposed_method_instance.epochs, greedy_mean_MAE[:-1],'y', label='Greedy sampling')
plt.legend(loc='best')
plt.xlabel('epoch')
plt.ylabel('MAE')
plt.title('Instance based QBC method MAE performance for {} dataset'.format(Method.dataset_name))
plt.savefig("../4.Results/{}/MAE/MAE".format(dataset))
plt.show()

# Append the average performance to the iteration performances
instance_total_R2, instance_total_MSE, instance_total_MAE  = np.append(proposed_method_instance.instance_R2, np.reshape(instance_mean_R2, (1,n_epochs+1)), axis=0), np.append(proposed_method_instance.instance_MSE, np.reshape(instance_mean_MSE, (1,n_epochs+1)), axis=0), np.append(proposed_method_instance.instance_MAE, np.reshape(instance_mean_MAE, (1,n_epochs+1)), axis=0)
upperbound_total_R2, upperbound_total_MSE, upperbound_total_MAE  = np.append(upperbound_method.upperbound_R2, np.reshape(upperbound_mean_R2, (1,n_epochs+1)), axis=0), np.append(upperbound_method.upperbound_MSE, np.reshape(upperbound_mean_MSE, (1,n_epochs+1)), axis=0), np.append(upperbound_method.upperbound_MAE, np.reshape(upperbound_mean_MAE, (1,n_epochs+1)), axis=0)
random_total_R2, random_total_MSE, random_total_MAE  = np.append(lowerbound_method.random_R2, np.reshape(random_mean_R2, (1,n_epochs+1)), axis=0), np.append(lowerbound_method.random_MSE, np.reshape(random_mean_MSE, (1,n_epochs+1)), axis=0), np.append(lowerbound_method.random_MAE, np.reshape(random_mean_MAE, (1,n_epochs+1)), axis=0)
greedy_total_R2, greedy_total_MSE, greedy_total_MAE  = np.append(baseline_method.baseline_R2, np.reshape(greedy_mean_R2, (1,n_epochs+1)), axis=0), np.append(baseline_method.baseline_MSE, np.reshape(greedy_mean_MSE, (1,n_epochs+1)), axis=0), np.append(baseline_method.baseline_MAE, np.reshape(greedy_mean_MAE, (1,n_epochs+1)), axis=0)

# Put the total performances in a dataframe and store them
cols = ['Epoch {}'.format(i+1) for i in range(n_epochs)].append('AUC')
rows = ['Iteration {}'.format(i+1) for i in range(Method.iterations)].append('Average')
instance_df_R2, instance_df_MSE, instance_df_MAE = pd.DataFrame(instance_total_R2, index=rows, columns=cols), pd.DataFrame(instance_total_MSE, index=rows, columns=cols), pd.DataFrame(instance_total_MAE, index=rows, columns=cols)
upperbound_df_R2, upperbound_df_MSE, upperbound_df_MAE = pd.DataFrame(upperbound_total_R2, index=rows, columns=cols), pd.DataFrame(upperbound_total_MSE, index=rows, columns=cols), pd.DataFrame(upperbound_total_MAE, index=rows, columns=cols)
random_df_R2, random_df_MSE, random_df_MAE = pd.DataFrame(random_total_R2, index=rows, columns=cols), pd.DataFrame(random_total_MSE, index=rows, columns=cols), pd.DataFrame(random_total_MAE, index=rows, columns=cols)
greedy_df_R2, greedy_df_MSE, greedy_df_MAE = pd.DataFrame(greedy_total_R2, index=rows, columns=cols), pd.DataFrame(greedy_total_MSE, index=rows, columns=cols), pd.DataFrame(greedy_total_MAE, index=rows, columns=cols)

instance_df_R2.to_csv("../4.Results/{}/R2/instance_R2.csv".format(dataset), header=cols), instance_df_MSE.to_csv("../4.Results/{}/MSE/instance_MSE.csv".format(dataset), header=cols), instance_df_MAE.to_csv("../4.Results/{}/MAE/instance_MAE.csv".format(dataset), header=cols) 
upperbound_df_R2.to_csv("../4.Results/{}/R2/upperbound_R2.csv".format(dataset), header=cols), upperbound_df_MSE.to_csv("../4.Results/{}/MSE/upperbound_MSE.csv".format(dataset), header=cols), upperbound_df_MAE.to_csv("../4.Results/{}/MAE/upperbound_MAE.csv".format(dataset), header=cols)
random_df_R2.to_csv("../4.Results/{}/R2/random_R2.csv".format(dataset), header=cols), random_df_MSE.to_csv("../4.Results/{}/MSE/random_MSE.csv".format(dataset), header=cols), random_df_MAE.to_csv("../4.Results/{}/MAE/random_MAE.csv".format(dataset), header=cols)
greedy_df_R2.to_csv("../4.Results/{}/R2/greedy_R2.csv".format(dataset), header=cols), greedy_df_MSE.to_csv("../4.Results/{}/MSE/greedy_MSE.csv".format(dataset), header=cols), greedy_df_MAE.to_csv("../4.Results/{}/MAE/greedy_MAE.csv".format(dataset), header=cols)

print("Instance based results:")
print(instance_total_R2)
print("Upperbound based results:")
print(upperbound_total_R2)
print("Random based results:")
print(random_total_R2)
print("Greedy based results:")
print(greedy_total_R2)