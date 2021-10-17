import pandas as pd
#Downgraded Arviz to 0.11.0
import pymc3 as pm
import arviz as az
import matplotlib.pyplot as plt
import theano.tensor as tt
import numpy as np
import bambi
import sklearn

# Use a theano shared variable to be able to exchange the data the model runs on
from theano import shared

from sklearn.model_selection import train_test_split

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



# CA Balance is a fixed effect
with pm.Model() as model:
    # Fixed slope - set prior for coefficient for ca balance
    ca_bal_coeff = pm.Normal('ca_bal_coeff', mu=4, sigma=1)
    
    prob_1 = pm.math.sigmoid(ca_bal_coeff*ca_balance)
     
    # Likelihood function
    sales = pm.Bernoulli('sales', p=prob_1, observed=targets)
    

#Set cores to 1
# Tuning samples will be discarded
with model:
    trace = pm.sample(1000,chains=2, tune=1000, cores=1,random_seed=1234)

summary = pm.summary(trace)
