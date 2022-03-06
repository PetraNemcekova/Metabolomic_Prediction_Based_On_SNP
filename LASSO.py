import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import functions as fcn
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn import linear_model

# # Loading preprocessed data
met1_preprocessed = pd.read_csv('met1_preprocessed.csv', index_col=0).T
gen1_preprocessed = pd.read_csv('gen1_preprocessed.csv', index_col=0).T

met1_arr, gen1_arr, gen1_feature_names = fcn.PrepareForPrediction(met1_preprocessed, gen1_preprocessed) 

# deviding into training and testing set

met1train, met1test, gen1train, gen1test = train_test_split(met1_arr, gen1_arr, test_size = 0.2,random_state=0)

# LASSO 

alphas = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
index = 0
j = 0

metabolites_results =  np.zeros((met1train.shape[1], 3 * len(alphas) )) # 'Alpha', 'MAE', 'MSE'

for alpha in alphas:

    for i in range(met1train.shape[1]):
        met = met1train[:,i]
        lasso = linear_model.Lasso(alpha=alpha)
        lasso.fit(gen1train, met) 
        met1_pred = lasso.predict(gen1test)

        mae = mean_absolute_error(met1test[:,i], met1_pred)
        mse = mean_squared_error(met1test[:,i], met1_pred)

        metabolites_results[i, j: j+3] = [alpha, mae, mse]

    index = index +1
    j = j+3

results_dataframe = pd.DataFrame(metabolites_results, columns=[['Alpha', 'MAE', 'MSE'] *  len(alphas)])
results_dataframe.to_csv('LASSO_02test_metabolites_results.csv')

