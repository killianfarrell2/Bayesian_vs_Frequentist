import pandas as pd
import numpy as np

import random

# Create list of targets with 10% Prior
targets = np.concatenate((np.zeros(9000),np.ones(1000))).reshape((10000, 1))
random.shuffle(targets)

# Generate a Normally distributed sample of ca balances
# mean is 1000 and sd is 500
ca_balance = np.random.normal(2000, 1000, 10000).reshape((10000, 1))
random.shuffle(ca_balance)

# Combine columns
data = pd.DataFrame(np.concatenate((ca_balance,targets),axis=1))
data.columns = ['ca_balance','sale']

import pystan
import matplotlib.pyplot as plt
X= np.random.normal(5,1,1000)
# Put data in correct format for stan
my_data = {'N':1000,'X':X}
# Compile model
sm = pystan.StanModel(file='C:\\KF_Repo\\Bayesian_vs_Frequentist\\my_model.stan')

