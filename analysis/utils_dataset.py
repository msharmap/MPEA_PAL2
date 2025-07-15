from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, random_split, DataLoader

import torch
import json
import pandas as pd
import numpy as np

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