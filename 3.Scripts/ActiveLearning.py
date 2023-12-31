""" The class for the general active learning method. """
import pandas as pd

class activelearning:
    n_trees = 100
    random_state = 0
    iterations = 5

    def __init__(self, dataset_name):
        self.dataset_name = dataset_name
        
    def data_read(self, dataset):
    # split the csv file in the input and target values
        data_path = "../2.Datasets/{}/{}.csv".format(self.dataset_name, dataset)
        df = pd.read_csv(data_path)

        # obtain the column names
        col_names = list(df.columns)
        target_length = 0

        for name in col_names: 
            if 'target' in name:
                target_length += 1

        target_names = col_names[-target_length:]

        inputs = list()
        targets = list()
        for i in range(len(df)):
            input_val = list()
            target_val = list()
            for col in col_names:
                if col in target_names:
                    target_val.append(df.loc[i, col])
                else:
                    input_val.append(df.loc[i, col])
            inputs.append(input_val)
            targets.append(target_val)

        n_instances = len(targets)
        return inputs, targets, n_instances, target_length
    
    def target_collect(self, targets, target_length):
    # collect all the values for specific targets in seperate lists
        targets_collected = list()
        for j, target in enumerate(targets):
            for i in range(target_length):
                # only make the seperate lists in the beginning
                if j == 0:
                    targets_collected.append(list())
                targets_collected[i].append(target[i])
        return targets_collected

    def instances_transfer(self, X_train, X_pool, y_train, y_pool, indices, method):
    # transfer data instances from the unlabelled pool to the training dataset
        instances_epoch = list()
        targets_epoch = list()

        for index in indices:
            instance, target = X_pool[index], y_pool[index]
            instances_epoch.append(instance)
            targets_epoch.append(target)

            X_train.append(instance)
            y_train.append(target)
            X_pool.pop(index)
            y_pool.pop(index)

        instances_epoch.append([])
        targets_epoch.append([])
        return instances_epoch, targets_epoch