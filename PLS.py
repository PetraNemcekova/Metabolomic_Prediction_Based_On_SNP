from sklearn.cross_decomposition import PLSRegression 
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

import numpy as np
import pandas as pd
import functions as fcn


# # Loading preprocessed data
met1_preprocessed = pd.read_csv('met1_preprocessed.csv', index_col=0).T
gen1_preprocessed = pd.read_csv('gen1_preprocessed.csv', index_col=0).T

met1_arr, gen1_arr, gen1_feature_names = fcn.PrepareForPrediction(met1_preprocessed, gen1_preprocessed) 

# deviding into training and testing set

met1train, met1test, gen1train, gen1test = train_test_split(met1_arr, gen1_arr, test_size = 0.4,random_state=0)

# Partial Least Squares Regression

metabolites_results =  np.zeros((met1train.shape[1], 2 )) # 'MAE', 'MSE'


for i in range(met1train.shape[1]):

    pls = PLSRegression(n_components=2)
    pls.fit(gen1train, met1train[:,i])
    met1_pred = pls.predict(gen1test)

    mse = mean_squared_error(met1test[:,i], met1_pred)
    mae = mean_absolute_error(met1test[:,i], met1_pred)
    metabolites_results[i,:] = [mae, mse]

results_dataframe = pd.DataFrame(metabolites_results, columns=['MAE', 'MSE'])
results_dataframe.to_csv('PLS_04test_results.csv')