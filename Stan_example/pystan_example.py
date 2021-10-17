import pystan
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random


# Create data set that is normally distributed around mean 5 sd 1
X= np.random.normal(5,1,10000)
# Put data in dictionary format for stan
my_data = {'N':10000,'X':X}

my_code = """
data {
      int N; // number of data points
      real X[N]; // data values
}

parameters {
    real mu; //mean
    real sigma; //standard deviation
}

model {
       X ~ normal(mu,sigma);
       
}
"""

# Create Model - this will help with recompilation issues
# When we want to run same code over again with new data - it will be much quicker
sm = pystan.StanModel(model_code=my_code)
# Call sampling function with data as argument
fit = sm.sampling(data=my_data, iter=2000, chains=4, seed=1,warmup=1000)

# Get summary statistics for parameters
print(fit)
# Extract generated samples
mu = fit.extract()['mu']

detailed_summary = fit.summary()

import arviz

# visual summary
fit.plot()
# Plot mu vs sigma
arviz.plot_joint(fit)

# Put Posterior draws into a dictionary
params = fit.extract()

# Dump model into dictionary
import pickle
with open('C:\\KF_Repo\\Bayesian_vs_Frequentist\\model_fit.pkl','wb') as f:
    pickle.dump({'model':sm,'fit':fit},f)
    
# Open model
with open('C:\\KF_Repo\\Bayesian_vs_Frequentist\\model_fit.pkl','rb') as f:
    data_dict = pickle.load(f)

fit = data_dict ['fit']
sm = data_dict ['model']
print(fit)

