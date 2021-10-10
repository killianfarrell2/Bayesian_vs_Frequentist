import pandas as pd
#Downgraded Arviz to 0.11.0
import pymc3 as pm
import arviz as az
import matplotlib.pyplot as plt
import theano.tensor as tt
import numpy as np
import sklearn

# Use a theano shared variable to be able to exchange the data the model runs on
from theano import shared

from sklearn.model_selection import train_test_split

import random

# Create list of targets with 10% Prior
targets = np.concatenate((np.zeros(9000),np.ones(1000)))
random.shuffle(targets)

# Generate a Normally distributed sample of ca balances
# mean is 1000 and sd is 500
ca_balance = np.random.normal(2000, 1000, 10000)
random.shuffle(ca_balance)


#Normal Distribution
#Changed to only have 1 standard deviation for all golfers
with pm.Model() as model:
    mean_ca_balance = pm.Normal('mean_ca_balance', mu=2000, sigma=1000)
    sd_ca_balance = pm.HalNormal('sd_ca_balance',sigma=100)
    
    # Observed scores to par follow normal distribution
    sales = pm.Normal("sales", mu=mean_ca_balance, sigma=sd_ca_balance, observed=targets)
    



#Set cores to 1
# Tuning samples will be discarded
with model:
    trace = pm.sample(1000,chains=2, tune=1000, cores=1,random_seed=1234)

# Set values for Model as test set for round 1
observed_golfers_shared.set_value(observed_golfers_test)

#Output is decimal number score to par - need to add to par and then round
num_samples = 10000
with model:
    pp_test_set = pm.sample_posterior_predictive(trace,samples=num_samples)
    


