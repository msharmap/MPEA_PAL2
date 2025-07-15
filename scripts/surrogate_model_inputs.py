# This file contaings surrogate model inputs
import numpy as np     
import os
import json
import math
import torch 

# User defined python classes and files
import sys
import utils_dataset as utilsd
import input_class 

np.random.seed(0)
torch.manual_seed(0)

# General inputs
run_folder = './'  # Folder where code is run and input json exist
num_run = 1
test_size = 0.01
output_folder = run_folder+'../bo_output/' # Folder where all outputs are stored
output_folder = output_folder+'debug_trial5/'
verbose = True
deep_verbose = False

# Reading and data processing inputs
add_target_noise = False
standardize_data = True

# Feature selection inputs
test_size_fs = 0.01
select_features_otherModels = False
select_features_NN = False

# BO inputs
n_trials = 5
n_batch = 10
n_update = 10
GP_0_BO = True
GP_L_BO = True
GP_NN_BO = True
random_seed = 'iteration'
maximization = True
new_values_predict_from_model = True

# Surrogate training boolean inputs
train_NN = True
saveModel_NN = True 
train_GP = True
predict_NN = False

# GP Model parameters
kernel = 'Matern'
learning_rate_gp0 = 0.01
learning_rate_gpL = 0.05
learning_rate_gpNN = 0.01

epochs_GP0 = 500
epochs_GPL = 500
epochs_GPNN = 500

# NN parameter
learning_rate = 1e-5
batch_size = 3
epochs = 500
l1 = 0.1 # 0.01
l2 = 0.4 # 0.004
num_nodes = 10
saveModel_filename = output_folder+'/NN.pt'
