import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
import pymc3 as pm


# Model 5: PYMC3 model

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

observed_occupations = X_train.occ_code.values - 1
observed_sale = y_train




with pm.Model() as model:
    # Intercept term for when there is no data for golfers
    intercept = pm.Normal('intercept', mu=0, sigma=1)
    occ_sd  = pm.Uniform('occ_sd',lower=0, upper=5)
    occ_mean = pm.Normal('occ_mean', mu=0, sigma=1)
    # golfer specific parameters
    occ_param = pm.Normal('occ_param', mu=occ_mean, sigma=occ_sd, shape=len(occupations))

    model_coefficients = intercept + occ_param[observed_occupations] 
    
    p = pm.Deterministic('p', pm.math.sigmoid(model_coefficients))
    
    # Observed scores Binomial Distribution
    sale =pm.Bernoulli("sale", p, observed=observed_sale)
    
    # Prior Predictive checks - generate samples without taking data
    prior_checks = pm.sample_prior_predictive(samples=1000, random_seed=1234)


#Set cores to 1
# Tuning samples will be discarded
with model:
    trace = pm.sample(1000,chains=2, tune=1000, cores=1,random_seed=1234)
























