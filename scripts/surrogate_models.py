# This file costructs surrogate models for the input datasets
import numpy as np     
import os
import json
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import math

# Torch specific module imports
import torch
import gpytorch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from torch.nn import functional as F

# botorch specific modules
from botorch.fit import fit_gpytorch_model
from botorch.models.gpytorch import GPyTorchModel

# Plotting libraries
import matplotlib as mpl
import matplotlib.pyplot as plt

# User defined python classes and files
import sys
sys.path.insert(0, '..')
sys.path.insert(0, '../feature_engineering/')

import utils_dataset as utilsd
import input_class 
import surrogate_model_inputs as model_input

np.random.seed(0)
torch.manual_seed(0)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

class NeuralNetwork(nn.Module):
    def __init__(self, in_features, num_nodes):
        super(NeuralNetwork,self).__init__()

        self.in_features = in_features
        
        # layer_list = torch.nn.ModuleList()
        
        # self.layer1 = nn.Linear(in_features, num_nodes,bias=True)
        # self.layer2 = nn.Linear(num_nodes,1,bias=True)
        self.layer1 = nn.Linear(in_features, num_nodes,bias=True)
        self.layer2 = nn.Linear(num_nodes, 100,bias=True)
        self.layer3 = nn.Linear(100,1,bias=True)
        
        # Define proportion or neurons to dropout
        self.dropout = nn.Dropout(0.1)  # Have to add to input
        
    def forward(self, x):
        layer1_out = torch.tanh(self.layer1(x))
        layer1_out = self.dropout(layer1_out)
        layer2_out = torch.tanh(self.layer2(layer1_out))
        layer2_out = self.dropout(layer2_out)
        output = torch.relu(self.layer3(layer2_out))
        # layer1_out = F.relu(self.layer1(x)) 
        # output = self.layer2(layer1_out)
        
        return output
    
class Train_NN():
    
    def __init__(self):
        if model_input.verbose:
            print('Starting training')
        
    def train_loop(self, dataloader, model, loss_fn, optimizer,lambda1,lambda2):
        
        size = len(dataloader.dataset)
        num_batches = len(dataloader)
        train_loss = 0.0
        l1_regularization, l2_regularization = 0.0, 0.0
        
        for batch, sample_batched in enumerate(dataloader):
            # Compute prediction and loss
            X = sample_batched['in_features']
            y = sample_batched['labels']
            # var = sample_batched['variance']
            pred = model(X)
            train_loss += loss_fn(pred, y).item()
            pred_loss = loss_fn(pred, y)
            
            all_linear1_params = torch.cat([x.view(-1) for x in model.layer1.parameters()])
            all_linear2_params = torch.cat([x.view(-1) for x in model.layer2.parameters()])
            all_linear3_params = torch.cat([x.view(-1) for x in model.layer3.parameters()])
            l1_regularization = lambda1 * (torch.norm(all_linear1_params, 1)
                                        +  torch.norm(all_linear2_params, 1)
                                        +  torch.norm(all_linear3_params, 1))
            l2_regularization = lambda2 * (torch.norm(all_linear1_params, 2)
                                        +  torch.norm(all_linear2_params, 2)
                                        +  torch.norm(all_linear3_params, 2))

            # l1_regularization = lambda1 * (torch.norm(all_linear1_params, 1)+torch.norm(all_linear2_params, 1))
            # l2_regularization = lambda2 * (torch.norm(all_linear1_params, 2)+torch.norm(all_linear2_params, 2))

            loss = pred_loss + l1_regularization + l2_regularization 

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        train_loss /=num_batches
        return train_loss


    def test_loop(self, dataloader, model, loss_fn):
        size = len(dataloader.dataset)
        num_batches = len(dataloader)
        test_loss, correct = 0, 0

        with torch.no_grad():
            for sample_batched in dataloader:
                X = sample_batched['in_features']
                y = sample_batched['labels']  
                # var = sample_batched['variance']
                pred = model(X)
                test_loss += loss_fn(pred, y).item()

        test_loss /= num_batches
        correct /= size
        return test_loss
    
    
# We will use the simplest form of GP model, exact inference
class ExactGPModel(gpytorch.models.ExactGP,GPyTorchModel):
    _num_outputs = 1  # to inform GPyTorchModel API
    MIN_INFERRED_NOISE_LEVEL = 1e-5
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        if model_input.kernel=='RBF':
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        elif model_input.kernel=='Matern':            
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=0.5))

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

# We will use the linear mean GP model, exact inference
class LinearGPModel(gpytorch.models.ExactGP,GPyTorchModel):
    _num_outputs = 1  # to inform GPyTorchModel API
    MIN_INFERRED_NOISE_LEVEL = 1e-5    
    def __init__(self, train_x, train_y, likelihood):
        super(LinearGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.LinearMean(train_x.shape[1], bias=True)
        if model_input.kernel=='RBF':
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        elif model_input.kernel=='Matern':            
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=0.5))
        
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

# We will use the NN mean GP model, exact inference
class NN_Gaussian(gpytorch.models.ExactGP,GPyTorchModel):
    _num_outputs = 1  # to inform GPyTorchModel API
    MIN_INFERRED_NOISE_LEVEL = 1e-5    
    def __init__(self, train_x, train_y, likelihood, saveModel_filename, num_nodes):
        super(NN_Gaussian, self).__init__(train_x, train_y, likelihood)
        self.mean_module = NeuralNetwork(train_x.shape[1],num_nodes)
        self.mean_module.load_state_dict(torch.load(saveModel_filename))
        self.mean_module.eval()
        if model_input.kernel=='RBF':
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        elif model_input.kernel=='Matern':            
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=0.5))
        
    def forward_stand_alone_fit_call(self,x):
        ''' 
        This forward call works when x is a 2-D tensor, 
        Like what is encountered in the initial fitting in BO
        or stand alone fitting
        ''' 
        output_nn = self.mean_module(x) 
        mean_x = torch.flatten(output_nn) 
        covar_x = self.covar_module(x)        
        
        return mean_x, covar_x

    def forward_acq_func_call(self,x):
        ''' 
        This forward call works when x is a 3-D tensor, 
        Like what is encountered when the acquisition function
        in BO calls the GP model
        ''' 
        output_nn = self.mean_module(x) 
        mean_x = output_nn 
        covar_x = self.covar_module(x)           
        
        return mean_x, covar_x
    
    def forward(self, x):
        ''' 
        This forward call works for the GP model, used to make predictions
        ''' 
        mean_x, covar_x = self.forward_stand_alone_fit_call(x)  
        if len(x.shape) == 2:
            mean_x, covar_x = self.forward_stand_alone_fit_call(x)  
        elif len(x.shape) == 3:
            mean_x, covar_x = self.forward_acq_func_call(x)
            mean_x = torch.reshape(mean_x,(mean_x.shape[0],mean_x.shape[1])) # Mean cannot be a 3-D tensor
            
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
    
    
def initial_train_surrogates(X_train, X_test, Y_train, Y_test, Var_train, Var_test, saveModel_filename):
    
    # NN parameters
    learning_rate =  model_input.learning_rate
    batch_size = model_input.batch_size
    epochs = model_input.epochs

    l1 = model_input.l1
    l2 =  model_input.l2
    
    # NN Model, Loss and Optimizer
    model = NeuralNetwork(X_train.shape[1],model_input.num_nodes)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    mse_gp0 = 0.0
    mse_gplinear = 0.0
    mse_gpnn = 0.0

    if (model_input.train_NN):
        # Dataloader for pytorch
        train_data = utilsd.InputDataset(X_train,Y_train,Var=Var_train)
        test_data = utilsd.InputDataset(X_test,Y_test,Var=Var_test)

        train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

        user_training = Train_NN()

        train_loss = []
        test_loss = []
        for t in range(epochs):
            train_loss_epoch = user_training.train_loop(train_dataloader, model, loss_fn, optimizer, l1, l2)
            test_loss_epoch = user_training.test_loop(test_dataloader, model, loss_fn)
            train_loss.append(train_loss_epoch)
            test_loss.append(test_loss_epoch)
            if ((t+1)%100 == 0 and model_input.verbose):
                print(f"Epoch {t+1}---> training error: {train_loss_epoch:>7f}, val error: {test_loss_epoch:>7f}")
        
        if model_input.deep_verbose:
            fig, ax = plt.subplots(figsize=(6,4))
            ax.plot(range(epochs),train_loss, label=f'Training Error,{train_loss_epoch:>7f}')
            ax.plot(range(epochs),test_loss, label=f'Validation Error,{test_loss_epoch:>7f}')
            ax.set_xlabel('Num. of epochs')
            ax.set_ylabel('MSE Loss')
            plt.legend()
        print("NN training Done!")

        if model_input.saveModel_NN:
            torch.save(model.state_dict(), saveModel_filename)
        
    if (model_input.predict_NN):
        mean_module = NeuralNetwork(X_train.shape[1],model_input.num_nodes)
        mean_module.load_state_dict(torch.load(saveModel_filename))
        mean_module.eval()
        
        output_nn = mean_module(X_train)
        
    if (model_input.train_GP):

        #--------------------------- GP-0 ---------------------------#
        # initialize likelihood and model
        print('GP-0 Model')
        training_iter = model_input.epochs_GP0
        likelihood_gp0 = gpytorch.likelihoods.GaussianLikelihood()
        model_gp0 = ExactGPModel(X_train, Y_train, likelihood_gp0)
    
        # Find optimal model hyperparameters
        model_gp0.train()
        likelihood_gp0.train()

        # Use the adam optimizer
        optimizer = torch.optim.Adam(model_gp0.parameters(), lr=model_input.learning_rate_gp0)  # Includes GaussianLikelihood parameters

        # "Loss" for GPs - the marginal log likelihood
        # mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood_gp0, model_gp0)
        mll = gpytorch.mlls.LeaveOneOutPseudoLikelihood(likelihood_gp0, model_gp0)

        for i in range(training_iter):
            optimizer.zero_grad()        # Zero gradients from previous iteration
            output = model_gp0(X_train)  # Output from model
            loss = -mll(output, Y_train) # Calc loss and backprop gradients            
            loss.backward()
            if model_input.deep_verbose:
                print('Iter %d/%d - Loss: %.3f  lengthscale: %.3f   noise: %.3f' % (
                    i + 1, training_iter, loss.item(),
                    model_gp0.covar_module.base_kernel.lengthscale.item(),
                    model_gp0.likelihood.noise.item()))            
            optimizer.step()        
        
        # Get into evaluation (predictive posterior) mode
        model_gp0.eval()
        likelihood_gp0.eval()

        # Make predictions by feeding model through likelihood
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            observed_mean = model_gp0(X_test)
            observed_pred = likelihood_gp0(model_gp0(X_test))
        num_points = Y_test.size()[1]
        mse_gp0 = 1.0/num_points*torch.sum(torch.square(Y_test - observed_mean.loc))

        
        #--------------------------- GP-Linear ---------------------------#
        # initialize likelihood and model
        print('GP-Linear Model') 
        training_iter = model_input.epochs_GPL
        likelihood_gpL = gpytorch.likelihoods.GaussianLikelihood()
        model_gpL = LinearGPModel(X_train, Y_train, likelihood_gpL)
        
        # Find optimal model hyperparameters - Check if we need this step
        model_gpL.train()
        likelihood_gpL.train()

        # Use the adam optimizer
        optimizer = torch.optim.Adam(model_gpL.parameters(), lr=model_input.learning_rate_gpL)  # Includes GaussianLikelihood parameters

        # "Loss" for GPs - the marginal log likelihood
        # mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood_gpL, model_gpL)
        mll = gpytorch.mlls.LeaveOneOutPseudoLikelihood(likelihood_gpL, model_gpL)
        
        for i in range(training_iter):
            optimizer.zero_grad()         # Zero gradients from previous iteration
            output = model_gpL(X_train)   # Output from model       
            loss = -mll(output, Y_train)  # Calc loss and backprop gradients 
            loss.backward()
            optimizer.step()
        
        # Get into evaluation (predictive posterior) mode
        model_gpL.eval()
        likelihood_gpL.eval()

        # Make predictions by feeding model through likelihood
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            observed_mean = model_gpL(X_test)
            observed_pred = likelihood_gpL(model(X_test))
        num_points = Y_test.size()[1]
        mse_gplinear = 1.0/num_points*torch.sum(torch.square(Y_test - observed_mean.loc))
        
        #--------------------------- GP-NN ---------------------------#
        # initialize likelihood and model
        print('GP-NN Model')
        training_iter = model_input.epochs_GPNN
        likelihood_gpnn = gpytorch.likelihoods.GaussianLikelihood()
        model_gpnn = NN_Gaussian(X_train, Y_train, likelihood_gpnn,
                                 saveModel_filename,model_input.num_nodes)

        # Find optimal model hyperparameters
        model_gpnn.train()
        likelihood_gpnn.train()

        # Use the adam optimizer
        optimizer = torch.optim.Adam(model_gpnn.parameters(), lr=model_input.learning_rate_gpNN)  # Includes GaussianLikelihood parameters

        # "Loss" for GPs - the marginal log likelihood
        # mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood_gpnn, model_gpnn)
        mll = gpytorch.mlls.LeaveOneOutPseudoLikelihood(likelihood_gpnn, model_gpnn)
        
        for i in range(training_iter):
            optimizer.zero_grad()         # Zero gradients from previous iteration
            output = model_gpnn(X_train)  # Output from model
            loss = -mll(output, Y_train)  # Calc loss and backprop gradients 
            loss.backward()
            
            if model_input.deep_verbose:
                print('Iter %d/%d - Loss: %.3f  lengthscale: %.3f   noise: %.3f' % (
                    i + 1, training_iter, loss.item(),
                    model_gpnn.covar_module.base_kernel.lengthscale.item(),
                    model_gpnn.likelihood.noise.item()))
            optimizer.step()
        
        # Get into evaluation (predictive posterior) mode
        model_gpnn.eval()
        likelihood_gpnn.eval()
        
        # Make predictions by feeding model through likelihood
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            observed_mean = model_gpnn(X_test)
            observed_pred = likelihood_gpnn(model(X_test))
        num_points = Y_test.size()[1]
        mse_gpnn = 1.0/num_points*torch.sum(torch.square(Y_test - observed_mean.loc))
        
        return mse_gp0, mse_gplinear, mse_gpnn, model_gp0, model_gpL, model_gpnn, likelihood_gp0, likelihood_gpL, likelihood_gpnn
    
    return

#--------------------------- NN ---------------------------#
def train_surrogate_NN(X_train,Y_train,saveModel_filename):
    
    # NN parameters
    learning_rate =  model_input.learning_rate
    batch_size = model_input.batch_size
    epochs = model_input.epochs

    l1 = model_input.l1
    l2 =  model_input.l2
    
    # NN Model, Loss and Optimizer
    model = NeuralNetwork(X_train.shape[1],model_input.num_nodes)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Dataloader for pytorch
    train_data = utilsd.InputDataset(X_train,Y_train)
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    user_training = Train_NN()

    train_loss = []
    for t in range(epochs):
        train_loss_epoch = user_training.train_loop(train_dataloader, model, loss_fn, optimizer, l1, l2)
        train_loss.append(train_loss_epoch)
        if ((t+1)%100 == 0 and model_input.verbose):
            print(f"Epoch {t+1}---> training error: {train_loss_epoch:>7f}")

    if model_input.saveModel_NN:
        torch.save(model.state_dict(), saveModel_filename)
            
    return
            
#--------------------------- GP-0 ---------------------------#
def train_surrogate_gp0(saveModel_filename,num_nodes,X_train,Y_train):
    
    mse_gp0 = 0.0 
    training_iter = model_input.epochs_GP0
    
    # initialize likelihood and model
    likelihood_gp0 = gpytorch.likelihoods.GaussianLikelihood()
    model_gp0 = ExactGPModel(X_train, Y_train, likelihood_gp0) 
    
    # Find optimal model hyperparameters
    model_gp0.train()
    likelihood_gp0.train()

    # Use the adam optimizer
    optimizer = torch.optim.Adam(model_gp0.parameters(), lr=model_input.learning_rate_gp0)  # Includes GaussianLikelihood parameters

    # "Loss" for GPs - the marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood_gp0, model_gp0)

    for i in range(training_iter):
        optimizer.zero_grad()        # Zero gradients from previous iteration
        output = model_gp0(X_train)  # Output from model
        loss = -mll(output, Y_train) # Calc loss and backprop gradients            
        loss.backward()
        optimizer.step()
        
    return model_gp0, likelihood_gp0

#--------------------------- GP-Linear ---------------------------#
def train_surrogate_gpL(saveModel_filename,num_nodes,X_train,Y_train):

    mse_gplinear = 0.0 
    training_iter = model_input.epochs_GPL
    
    # initialize likelihood and model
    likelihood_gpL = gpytorch.likelihoods.GaussianLikelihood()
    model_gpL = LinearGPModel(X_train, Y_train, likelihood_gpL)
    
    # Find optimal model hyperparameters 
    model_gpL.train()
    likelihood_gpL.train()

    # Use the adam optimizer
    optimizer = torch.optim.Adam(model_gpL.parameters(), lr=model_input.learning_rate_gpL)  # Includes GaussianLikelihood parameters

    # "Loss" for GPs - the marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood_gpL, model_gpL)

    for i in range(training_iter):
        optimizer.zero_grad()         # Zero gradients from previous iteration
        output = model_gpL(X_train)   # Output from model       
        loss = -mll(output, Y_train)  # Calc loss and backprop gradients 
        loss.backward()
        optimizer.step()
        
    return model_gpL, likelihood_gpL


#--------------------------- GP-NN ---------------------------#
def train_surrogate_gpnn(saveModel_filename,num_nodes,X_train,Y_train):

    mse_gpnn = 0.0
    training_iter = model_input.epochs_GPNN
    
    # Initialize likelihood and model
    likelihood_gpnn = gpytorch.likelihoods.GaussianLikelihood()
    model_gpnn = NN_Gaussian(X_train, Y_train, likelihood_gpnn, saveModel_filename,num_nodes)
    
    # Find optimal model hyperparameters
    model_gpnn.train()
    likelihood_gpnn.train()

    # Use the adam optimizer
    optimizer = torch.optim.Adam(model_gpnn.parameters(), lr=model_input.learning_rate_gpNN)  # Includes GaussianLikelihood parameters

    # "Loss" for GPs - the marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood_gpnn, model_gpnn)

    for i in range(training_iter):
        optimizer.zero_grad()         # Zero gradients from previous iteration
        output = model_gpnn(X_train)  # Output from model
        loss = -mll(output, Y_train)  # Calc loss and backprop gradients 
        loss.backward()
        optimizer.step()
        
    return model_gpnn, likelihood_gpnn

def predict_surrogates(model, likelihood, X):
    
    # Get into evaluation (predictive posterior) mode
    model.eval()
    likelihood.eval()

    # Make predictions by feeding model through likelihood
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        prediction = model(X)
        prediction = likelihood(model(X))

    observed_mean = prediction.mean
    observed_var = prediction.variance
    observed_covar = prediction.covariance_matrix

    return observed_mean, observed_var
    
    
