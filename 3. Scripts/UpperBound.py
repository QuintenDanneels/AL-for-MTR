""" The subclass for the upper bound method. """
from ActiveLearning import *
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, auc

class upperbound(activelearning):
    def __init__(self, n_epochs):
        self.n_epochs = n_epochs
        self.epochs = [i for i in range(n_epochs)]
        self.upperbound_R2 = np.zeros([self.iterations, self.n_epochs+1])
        self.upperbound_MSE = np.zeros([self.iterations, self.n_epochs+1])
        self.upperbound_MAE = np.zeros([self.iterations, self.n_epochs+1])
    
    def training(self, X_rest, X_test, y_rest, y_test, target_length): 
        R2 = np.zeros([1, self.n_epochs+1])
        MSE = np.zeros([1, self.n_epochs+1])
        MAE = np.zeros([1, self.n_epochs+1])
        Y_pred = np.zeros([len(X_test), target_length])

        # first, collect the values of all the instances for each target feature
        y_rest_coll = self.target_collect(y_rest, target_length)
    
        y_test_preds = np.zeros(shape=(len(X_test), target_length))
        for i in range(target_length):
            # next, fit a model to the training + labeled pool set
            rf_regressor = RandomForestRegressor(n_estimators=self.n_trees, random_state=self.random_state)  # default value for max features is 1.0 => N features
            rf_regressor.fit(X_rest, y_rest_coll[i])
            
            # next, predict the target values of the test set
            test_preds = rf_regressor.predict(X_test)
            y_test_preds[:,i] = np.round(test_preds, 5)
        Y_pred[:,:] = y_test_preds
        r2, mse, mae = (np.round(r2_score(np.asarray(y_test), y_test_preds), 4)), (np.round(mean_squared_error(np.asarray(y_test), y_test_preds), 4)), (np.round(mean_absolute_error(np.asarray(y_test), y_test_preds), 4))
        for i in range(self.n_epochs):
            R2[:,i] = (r2)
            MSE[:,i] = (mse)
            MAE[:,i] = (mae)
        r2_auc, mse_auc, mae_auc = np.round(auc(self.epochs, R2[0,:-1]), 4), np.round(auc(self.epochs, MSE[0,:-1]), 4), np.round(auc(self.epochs, MAE[0,:-1]), 4) 
        R2[:,-1] = (r2_auc)
        MSE[:,-1] = (mse_auc)
        MAE[:,-1] = (mae_auc)

        cols = ["Target_{}".format(i+1) for i in range(target_length)]
        Y_pred_df = pd.DataFrame(Y_pred, columns=cols)
        return R2, MSE, MAE, Y_pred_df