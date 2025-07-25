from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, random_split, DataLoader

import torch
import json
import pandas as pd
import numpy as np

# User defined python classes and files
import sys
sys.path.insert(0, './feature_engineering/')

import utils_dataset as utilsd
import input_class 
import surrogate_model_inputs as model_input
import feature_selection_methods as feature_selection

# Add slicing of the input XX tensor with additional input for the columns picked out by XGBoost or other feature selection methods
class InputDataset(Dataset):
    """ Input dataset used for training """

    def __init__(self, XX, YY, Var=None, transform=None):
        """
        Args:
            XX: NN Input features vector as a torch tensor
            YY: NN Labels vector as a torch tensor
            descriptors(list of strings): Names of the input features
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.XX = XX
        self.YY = YY
        self.var = Var
        self.transform = transform
        
    def __len__(self):
        return self.XX.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        x = self.XX[idx,:]
        y = self.YY[:,idx]
        if self.var != None:
            var = self.var[idx]
            item = {'in_features':x,'labels':y,'variance':var}
        else:
            item = {'in_features':x,'labels':y}

        return item
    

def standardize_data(x):
    scalerX = StandardScaler().fit(x)
    x_train = scalerX.transform(x)
    return x_train, scalerX
    
def standardize_test_data(x,scalerX):
    x_test = scalerX.transform(x)
    return x_test

def generate_training_data(random_state,test_size):
    
    # Reading the input json file with dataset filename and path information
    with open(model_input.run_folder+'inputs.json', "r") as f:
        input_dict = json.load(f)

    input_type = input_dict['InputType']
    input_path = input_dict['InputPath']
    input_file = input_dict['InputFile']
    add_target_noise = input_dict['AddTargetNoise']
    
    input = input_class.inputs(input_type=input_type,
                               input_path=input_path,
                               input_file=input_file,
                               add_target_noise=add_target_noise)
    
    XX, YY, descriptors = input.read_inputs(model_input.verbose)
    
    # Transforming datasets by standardization
    if model_input.standardize_data:
        X_stand, scalerX_transform = utilsd.standardize_data(XX)
        Y_stand, scalerY_transform = utilsd.standardize_data(YY)
    else:
        X_stand=XX.to_numpy()
        Y_stand = YY
    
    # Checking if we should use xgboost recommended descriptors or all descriptors
    if model_input.select_features_otherModels:
        fs = feature_selection.feature_selection_algorithms(X_stand,Y_stand,
                                                            test_size=model_input.test_size_fs,
                                                            random_state=random_state)
        xg_boost_descriptors = fs.selected_features_xgboost(descriptors)
        if model_input.verbose:
            print('Selected Features, ', xg_boost_descriptors)
    else:
        xg_boost_descriptors = descriptors
      
    XX = pd.DataFrame(XX, columns=xg_boost_descriptors)
    if model_input.standardize_data:
        X_stand, scalerX_transform = utilsd.standardize_data(XX)
    else:
        X_stand=XX.to_numpy()
        
    # Creating train-test split in data
    X_train, X_test, Y_train, Y_test = train_test_split(X_stand, Y_stand, 
                                                        test_size=test_size, 
                                                        random_state=random_state) #,stratify=Y_stand)
   
    Var_train = torch.ones(len(Y_train)) 
    Var_test = torch.ones(len(Y_test)) 

    # Converting data arrays to torch tensors
    X_train = torch.tensor(X_train).to(torch.float32)
    Y_train = np.transpose(Y_train) # Ytrain has to have only one row for GP training
    Y_train = torch.tensor(Y_train).to(torch.float32)

    X_test = torch.tensor(X_test).to(torch.float32)
    Y_test = np.transpose(Y_test) # Ytrain has to have only one row for GP training
    Y_test = torch.tensor(Y_test).to(torch.float32)
    
    if model_input.standardize_data:
        return X_train, X_test, Y_train, Y_test, Var_train, Var_test, scalerX_transform, scalerY_transform
    else:
        return X_train, X_test, Y_train, Y_test, Var_train, Var_test

def generate_training_data_NN(random_state,test_size):
    
    # Reading the input json file with dataset filename and path information
    with open(model_input.run_folder+'inputs.json', "r") as f:
        input_dict = json.load(f)

    input_type = input_dict['InputType']
    input_path = input_dict['InputPath']
    input_file = input_dict['InputFile']
    add_target_noise = input_dict['AddTargetNoise']
    
    input = input_class.inputs(input_type=input_type,
                               input_path=input_path,
                               input_file=input_file,
                               add_target_noise=add_target_noise)
    
    XX, YY, descriptors = input.read_inputs(model_input.verbose)
    
    # Transforming datasets by standardization
    if model_input.standardize_data:
        X_stand, scalerX_transform = utilsd.standardize_data(XX)
        Y_stand, scalerY_transform = utilsd.standardize_data(YY)
    else:
        X_stand=XX.to_numpy()
        Y_stand = YY
    
    # Checking if we should use xgboost recommended descriptors or all descriptors    
    if model_input.select_features_NN:
        fs = feature_selection.feature_selection_algorithms(X_stand,Y_stand,
                                                            test_size=model_input.test_size_fs,
                                                            random_state=random_state)
        xg_boost_descriptors = fs.selected_features_xgboost(descriptors)
        if model_input.verbose:
            print('Selected Features, ', xg_boost_descriptors)
    else:
        xg_boost_descriptors = descriptors
    
    XX = pd.DataFrame(XX, columns=xg_boost_descriptors)
    if model_input.standardize_data:
        X_stand, scalerX_transform = utilsd.standardize_data(XX)
    else:
        X_stand=XX.to_numpy()
 
    # Creating train-test split in data
    X_train, X_test, Y_train, Y_test = train_test_split(X_stand, Y_stand, 
                                                        test_size=model_input.test_size, 
                                                        random_state=random_state)
   
    Var_train = torch.ones(len(Y_train)) 
    Var_test = torch.ones(len(Y_test)) 

    # Converting data arrays to torch tensors
    X_train = torch.tensor(X_train).to(torch.float32)
    Y_train = np.transpose(Y_train) # Ytrain has to have only one row for GP training
    Y_train = torch.tensor(Y_train).to(torch.float32)

    X_test = torch.tensor(X_test).to(torch.float32)
    Y_test = np.transpose(Y_test) # Ytrain has to have only one row for GP training
    Y_test = torch.tensor(Y_test).to(torch.float32)
    
    if model_input.standardize_data:
        return X_train, X_test, Y_train, Y_test, Var_train, Var_test, scalerX_transform, scalerY_transform
    else:
        return X_train, X_test, Y_train, Y_test, Var_train, Var_test


def read_training_data(random_state):
    
    # Reading the input json file with dataset filename and path information
    with open(model_input.run_folder+'inputs_training.json', "r") as f:
        input_dict = json.load(f)

    input_type = input_dict['InputType']
    input_path = input_dict['InputPath']
    input_file = input_dict['InputFile']
    add_target_noise = input_dict['AddTargetNoise']
    
    input = input_class.inputs(input_type=input_type,
                               input_path=input_path,
                               input_file=input_file,
                               add_target_noise=add_target_noise)
    
    XX, YY, descriptors = input.read_inputs(model_input.verbose)
    
    # Transforming datasets by standardization
    if model_input.standardize_data:
        X_stand, scalerX_transform = utilsd.standardize_data(XX)
        Y_stand, scalerY_transform = utilsd.standardize_data(YY)
    else:
        X_stand=XX.to_numpy()
        Y_stand = YY
    
    # Checking if we should use xgboost recommended descriptors or all descriptors
    if model_input.select_features_otherModels:
        fs = feature_selection.feature_selection_algorithms(X_stand,Y_stand,
                                                            test_size=model_input.test_size_fs,
                                                            random_state=random_state)
        xg_boost_descriptors = fs.selected_features_xgboost(descriptors,onlyImportant=True)
        if model_input.verbose:
            print('Selected Features, ', xg_boost_descriptors)
    else:
        xg_boost_descriptors = descriptors
      
    XX = pd.DataFrame(XX, columns=xg_boost_descriptors)
    if model_input.standardize_data:
        X_stand, scalerX_transform = utilsd.standardize_data(XX)
    else:
        X_stand=XX.to_numpy()
        
    # Creating training data
    X_train, Y_train = X_stand, Y_stand
    Var_train = torch.ones(len(Y_train)) 

    # Converting data arrays to torch tensors
    X_train = torch.tensor(X_train).to(torch.float32)
    Y_train = np.transpose(Y_train) # Ytrain has to have only one row for GP training
    Y_train = torch.tensor(Y_train).to(torch.float32)

    if model_input.standardize_data:
        return X_train, Y_train, Var_train, scalerX_transform, scalerY_transform, xg_boost_descriptors
    else:
        return X_train, Y_train, Var_train, xg_boost_descriptors


def read_test_data(random_state, xg_boost_descriptors, scalerX_transform):
    
    # Reading the input json file with dataset filename and path information
    with open(model_input.run_folder+'inputs_testing.json', "r") as f:
        input_dict = json.load(f)

    input_type = input_dict['InputType']
    input_path = input_dict['InputPath']
    input_file = input_dict['InputFile']
    add_target_noise = input_dict['AddTargetNoise']
    
    input = input_class.inputs(input_type=input_type,
                               input_path=input_path,
                               input_file=input_file,
                               add_target_noise=add_target_noise)
    
    XX, YY, descriptors = input.read_inputs(model_input.verbose)
    
    # Transforming datasets by standardization
    Y_stand = YY
    
    # Keeping only the descriptors selected using the training data 
    XX = pd.DataFrame(XX, columns=xg_boost_descriptors)
    if model_input.standardize_data:
        X_stand = utilsd.standardize_test_data(XX, scalerX_transform)
    else:
        X_stand=XX.to_numpy()
        
    # Creating test data
    X_test, Y_test = X_stand, Y_stand
    Var_test = torch.ones(len(Y_test)) 

    # Converting data arrays to torch tensors
    X_test = torch.tensor(X_test).to(torch.float32)
    Y_test = np.transpose(Y_test) # Ytrain has to have only one row for GP training
    Y_test = torch.tensor(Y_test).to(torch.float32)
    
    return X_test, Y_test, Var_test