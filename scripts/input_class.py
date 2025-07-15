import sklearn
import numpy as np     
import csv 
import copy 
import random 
import pandas as pd
import pickle
import json  
import openpyxl
import itertools

# User defined files and classes
import feature_selection_methods as feature_selection
import utils_dataset as utilsd

# Plotting libraries
import matplotlib as mpl
import matplotlib.pyplot as plt

# Tick parameters
plt.rcParams['xtick.labelsize'] = 15
plt.rcParams['ytick.labelsize'] = 15
plt.rcParams['xtick.major.size'] = 5
plt.rcParams['xtick.major.width'] = 1
plt.rcParams['xtick.minor.size'] = 5
plt.rcParams['xtick.minor.width'] = 1
plt.rcParams['ytick.major.size'] = 5
plt.rcParams['ytick.major.width'] = 1
plt.rcParams['ytick.minor.size'] = 5
plt.rcParams['ytick.minor.width'] = 1

plt.rcParams['axes.labelsize'] = 20
plt.rcParams['axes.titlesize'] = 20
plt.rcParams['legend.fontsize'] = 15


class inputs:
    def __init__(self,input_type='MPEA',input_path='../datasets/',input_file='curated_MPEA_initial_training_hardness_value.csv',add_target_noise = False, composition_MPEA = False):
        self.input_type = input_type
        self.input_path = input_path
        self.input_file = input_file
        self.add_target_noise = add_target_noise
        self.composition_MPEA = composition_MPEA
        self.filename   = self.input_path + self.input_file
        
    def read_inputs(self, verbose):
        if verbose:
            print('Reading data for the input dataset type: ', self.input_type)
        
        # Add options for different datasets that we want to read       
        if self.input_type == 'MPEA':
            XX, YY, descriptors = self.read_MPEA()          
        return XX, YY, descriptors    
    
    def read_MPEA(self):
        '''
        This function reads the dataset from the HEA review paper: https://www.nature.com/articles/s41597-020-00768-9
        input_type='MPEA',
        input_path='../datasets/',
        input_file='curated_MPEA_initial_training_hardness_value.csv'
        '''     
        data = pd.read_csv(self.filename)
 
        input_composition_cols = data.columns[0:29]
        input_property_cols = data.columns[30:35]
        input_composition_df = pd.DataFrame(data, columns=['Ti', 'Pd', 'Ga', 'Al', 'Co', 'Si', 'Mo', 'Sc', 'Zn', 'C', 'Sn', 'Nb', 'Ag', 'Mg', 'Mn', 'Y', 
                                    'Re', 'W', 'Zr', 'Ta', 'Fe', 'Cr', 'B', 'Cu', 'Hf', 'Li', 'V', 'Nd', 'Ni', 'Ca'])

        if self.composition_MPEA:
            XX = pd.DataFrame(data, columns=input_composition_cols)
            descriptors = input_composition_cols
        else:
            XX = pd.DataFrame(data, columns=input_property_cols)
            descriptors = input_property_cols
        target = copy.deepcopy(data['Target'].to_numpy())
        YY = target.reshape(-1,1)

        return XX, YY, descriptors


if __name__=="__main__":
    
    run_folder = './.'
    with open(run_folder+'inputs.json', "r") as f:
        input_dict = json.load(f)
    
    input_type = input_dict['InputType']
    input_path = input_dict['InputPath']
    input_file = input_dict['InputFile']
    
    input=inputs()
    XX, YY, descriptors = input.read_inputs()
    
    X_stand = utilsd.standardize_data(XX)
    Y_stand = utilsd.standardize_data(YY)
        
