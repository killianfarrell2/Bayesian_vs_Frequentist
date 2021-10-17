import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
import pystan


# Model 3: Stan Hierarchical Bayesain Model (Occupation codes drawn from same distr)
# Set Prior to 50% using odds ration and logistic function

# Import Sales data
sales_data = pd.read_csv('C:\\KF_Repo\\Bayesian_vs_Frequentist\\Pensions_example\\Data\\Pension_sales_3.csv') 

X = pd.DataFrame(sales_data['Occupation'].values.reshape(len(sales_data), 1))
y= sales_data['Sale'].values.reshape(len(sales_data), 1)

# Create occupation reference table
occupations = pd.DataFrame(np.unique(X.values))
occupations['occ_code'] = np.arange(len(occupations)) + 1

# Map back to X dataframe
X['occ_code']=X.merge(occupations,how='left').values[:,1]

# Drop original column
X = X.drop(columns=[0])
# Convert back to int
X['occ_code'] = X['occ_code'].astype(str).astype(int)

# Split into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,stratify=y, random_state=1234)

# Reshape arrays so they can be taken by Stan
y_train = y_train.reshape(len(y_train),)
X_train = X_train.values.reshape(len(X_train),)

# Put data in dictionary format for stan
my_data = {'N':len(X_train),'n_occ':len(np.unique(X_train)),'y':y_train,'X':X_train}

my_code = """
data {
      int N; // number of data points
      int<lower=0> n_occ; //number of occupations
      int<lower=0,upper=1> y[N];// data values
      int<lower=0> X[N]; //occupation codes data values
}

parameters {
    
        //hyper parameters
        real mu_occ;
        // Standard deviation on occupation code has lower bound of 0
        real <lower=0> sd_occ;
    
        // Parameters in logistic regression
        real intercept; // intercept term in model
        vector[n_occ] occ; //Coefficient for each occupation
       
}

model {
       //hyper priors - set a wide group
       mu_occ ~ normal(0,3);
       sd_occ ~ normal(0,3);
       // Set narrow prior on intercept to be 50%
       intercept ~ normal(0,0.01);
       
       //priors
       occ ~ normal(mu_occ, sd_occ);
       
       //likelihood
       y ~ bernoulli_logit(intercept + occ[X]);
     
}
"""

# Create Model - this will help with recompilation issues
stan_model = pystan.StanModel(model_code=my_code)

# Call sampling function with data as argument
fit = stan_model.sampling(data=my_data, iter=2000, chains=4, seed=1,warmup=1000)

# Get summary statistics for parameters
print(fit)
# Extract generated samples
occ = fit.extract()['occ']

detailed_summary = fit.summary()
