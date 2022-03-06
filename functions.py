import os
import numpy as np
import os

from pandas.core.frame import DataFrame


def replaceColumnsWith1stRowValues (dataframe):
    
    result = dataframe
    # fix headers with first row (names was previously in first column)
    result.columns = result.iloc[0]
    # drop first row
    result = result.iloc[1: , :]
    return result

def replaceIndexWith1stColumnValues(dataframe):
    
    result = dataframe
    # fix index/rows with first column data
    result.index = result.iloc[:, 0].values
    firstColumnName = result.columns.values[0]
    # drop first column
    result = result.drop([firstColumnName], axis = 1)
    return result

# import csv
def getFilePath (relativePath):
    return os.path.join(os.path.dirname(__file__), relativePath)

def getInterselectedWithFamilies(metabolome_dataframe,genome_dataframe):

    # fix headers with first row

    genome_families_list = genome_dataframe.columns.values.tolist()
    metabolome_families_list = metabolome_dataframe.columns.values.tolist()

    metabolome_families_intersected = np.intersect1d(metabolome_families_list, genome_families_list)

    genome_dataframe_intersected = genome_dataframe[metabolome_families_intersected]
    genome_dataframe_intersected = genome_dataframe_intersected.loc[:,~genome_dataframe_intersected.columns.duplicated()]
    metabolome_dataframe_intersected = metabolome_dataframe[metabolome_families_intersected]
    metabolome_dataframe_intersected = metabolome_dataframe_intersected.loc[:,~metabolome_dataframe_intersected.columns.duplicated()]

    return  metabolome_dataframe_intersected, genome_dataframe_intersected

def deleteGenomeFeaturesWithNaN (data, features):
    data_preprocessed = data

    for i in range(len(features)):
        if (data.iloc[i,:].isna().sum() > 0):
            unused_features = features.iloc[i]
            data_preprocessed = data_preprocessed.drop(unused_features)

    return data_preprocessed

def deleteMetabolomeFeaturesWithNaN (data, features):
    data_preprocessed = data

    for i in range(len(features)):
        if (data.iloc[i,:].isna().sum() > 0):
            unused_features = features[i]
            data_preprocessed = data_preprocessed.drop(unused_features)

    return data_preprocessed


def standardizeValues(dataFrame: DataFrame) -> DataFrame:
    result = dataFrame.replace(2,-1)
    return result

def PrepareForPrediction(metabolomic, genomic):
    standardized1 = standardizeValues(genomic)
    
    # Shuffle
    
    perm = np.random.permutation(metabolomic.shape[0])

    metabolome1_mixed = metabolomic.iloc[perm,:]
    genome1_mixed = standardized1.iloc[perm,:]

    # To array

    met1_arr = metabolome1_mixed.values
    gen1_arr = genome1_mixed.values

    gen1_feature_names = list(genome1_mixed.columns)

    return  met1_arr, gen1_arr, gen1_feature_names