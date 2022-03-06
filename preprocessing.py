import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import functions as fcn
import math
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor

# Loading data

# genome = pd.read_csv('..\Data\B_IBS.csv')
# genome = pd.read_csv('..\Data\C_IBS_imputed_MNI.csv')
# genome = pd.read_csv('..\Data\D_IBD_imputed_flanking.csv')
genome = pd.read_csv('..\Data\E_IBS_imputed_flanking.csv')

metabolome1 = pd.read_csv('..\Data\Metabolome_day1.csv')
metabolome2 = pd.read_csv('..\Data\Metabolome_day2.csv')

genome_markers = genome['marker']

metabolites1_names = metabolome1.columns[1:]
metabolites2_names = metabolome2.columns[1:]


## Preprocessing

# removing not matching data

metabolome1_dataframe_intersected,genome1_dataframe_intersected = fcn.getInterselectedWithFamilies(fcn.replaceColumnsWith1stRowValues(metabolome1.T), fcn.replaceIndexWith1stColumnValues(genome) )
metabolome2_dataframe_intersected,genome2_dataframe_intersected = fcn.getInterselectedWithFamilies(fcn.replaceColumnsWith1stRowValues(metabolome2.T), fcn.replaceIndexWith1stColumnValues(genome) )

# remove markers with nan values

met1_preprocessed = fcn.deleteMetabolomeFeaturesWithNaN(metabolome1_dataframe_intersected, metabolites1_names)
met2_preprocessed = fcn.deleteMetabolomeFeaturesWithNaN(metabolome2_dataframe_intersected, metabolites2_names)
gen1_preprocessed = fcn.deleteGenomeFeaturesWithNaN(genome1_dataframe_intersected, genome_markers)
gen2_preprocessed = fcn.deleteGenomeFeaturesWithNaN(genome2_dataframe_intersected, genome_markers)

# export preprocessed data

# met1_preprocessed.to_csv('B_IBS_met1_preprocessed.csv')
# met2_preprocessed.to_csv('B_IBS_met2_preprocessed.csv')
# gen1_preprocessed.to_csv('B_IBS_gen1_preprocessed.csv')
# gen2_preprocessed.to_csv('B_IBS_gen2_preprocessed.csv')

# met1_preprocessed.to_csv('C_IBS_met1_preprocessed.csv')
# met2_preprocessed.to_csv('C_IBS_met2_preprocessed.csv')
# gen1_preprocessed.to_csv('C_IBS_gen1_preprocessed.csv')
# gen2_preprocessed.to_csv('C_IBS_gen2_preprocessed.csv')

# met1_preprocessed.to_csv('D_IBD_met1_preprocessed.csv')
# met2_preprocessed.to_csv('D_IBD_met2_preprocessed.csv')
# gen1_preprocessed.to_csv('D_IBD_gen1_preprocessed.csv')
# gen2_preprocessed.to_csv('D_IBD_gen2_preprocessed.csv')

met1_preprocessed.to_csv('E_IBS_met1_preprocessed.csv')
met2_preprocessed.to_csv('E_IBS_met2_preprocessed.csv')
gen1_preprocessed.to_csv('E_IBS_gen1_preprocessed.csv')
gen2_preprocessed.to_csv('E_IBS_gen2_preprocessed.csv')

