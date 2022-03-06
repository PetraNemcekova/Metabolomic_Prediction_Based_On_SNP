import numpy as np
import pandas as pd
import functions as fcn
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error


# # Loading preprocessed data
met1_preprocessed = pd.read_csv('met1_preprocessed.csv', index_col=0).T
gen1_preprocessed = pd.read_csv('gen1_preprocessed.csv', index_col=0).T

met1_arr, gen1_arr, gen1_feature_names = fcn.PrepareForPrediction(met1_preprocessed, gen1_preprocessed) 

# deviding into training and testing set

met1train, met1test, gen1train, gen1test = train_test_split(met1_arr, gen1_arr, test_size = 0.3,random_state=0)

# Partial Least Squares Regression

# index = 0
j=0

epsilons = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
C_params = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]

metabolites_results =  np.zeros((met1train.shape[1], 4 * len(epsilons) * len(C_params))) # 'Epsilons', 'C_params', 'MAE', 'MSE', 'Train Score'

for epsilon in epsilons:

    for C_param in C_params:

        for i in range(met1train.shape[1]):
            met = met1train[:,i]

            # mySVR = make_pipeline(StandardScaler(), SVR(C=C_param, epsilon=epsilon))
            # mySVR.fit(gen1train, met)

            mySVR = SVR(C = C_param, epsilon = epsilon)
            mySVR.fit(gen1train, met)
            met1_pred=mySVR.predict(gen1test)

            mae =  mean_absolute_error(met1test[:,i], met1_pred)
            mse = mean_squared_error(met1test[:,i], met1_pred)
            metabolites_results[i,j:j+4] = [epsilon, C_param, mae, mse]

        # index = index+1 
        j = j+4

results_dataframe = pd.DataFrame(metabolites_results, columns=[['Epsilon', 'C_param', 'MAE', 'MSE']* (len(epsilons) * len(C_params))])
results_dataframe.to_csv('SVR_03test_metabolites_results.csv')




