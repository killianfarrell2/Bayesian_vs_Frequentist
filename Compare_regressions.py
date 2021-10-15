import pystan
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing


# Model 1: Intercept and one coefficient for all occupation codes
# Create list of targets with 10% Prior
targets = np.concatenate((np.ones(1000),np.zeros(9000))).reshape(10000, 1)

# Create label column y for train and test
lb = preprocessing.LabelBinarizer()
y = lb.fit_transform(targets)

# Create categorical data (occupation codes) 20 in total
occupation_codes = np.random.randint(20, size=(10000)).reshape((10000, 1))

# Combine columns
X = pd.DataFrame(occupation_codes)
X.columns = ['Occ_code']

# Split into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,stratify=y, random_state=1234)

# Add constant to X-train for statsmodels
# Adding constant removes affect of other variables
X_train_2 = sm.add_constant(X_train)
X_test_2 = sm.add_constant(X_test)

# building the model and fitting the data
log_reg = sm.Logit(y_train, X_train_2).fit()

# Print Summary
print(log_reg.summary())

# performing predictions on the test datdaset
y_hat = np.array(log_reg.predict(X_test_2))

# Print performance metrics on test set

auroc_sm = roc_auc_score(y_test, y_hat)

if auroc_sm >= 0.5:
    print('sm auroc: ',auroc_sm)
else:
    print('sm auroc: ',(1-auroc_sm))

auprc_sm = average_precision_score(y_test, y_hat)
print('sm auprc: ',auprc_sm)

# Create logistic regression model using scikit learn

logreg_sk = LogisticRegression()
logreg_sk.fit(X_train, y_train.ravel())

logreg_sk.classes_

# Predict probabilities as predicted labels will all be 0
y_pred_sk_p = logreg_sk.predict_proba(X_test)
y_1 = y_pred_sk_p[:,1]
y_0 = y_pred_sk_p[:,0]

logit_roc_auc = roc_auc_score(y_test, y_1)

if logit_roc_auc >= 0.5:
    print('sk auroc_1: ',logit_roc_auc)
else:
    print('sk auroc: ',(1-logit_roc_auc))

auprc = average_precision_score(y_test, y_0)
print('sk auprc: ',auprc)

# Coefficients are very similar when constant added to statsmodels
print('sk coefficients: ',logreg_sk.coef_, logreg_sk.intercept_)

# Results for auroc and auprc are almost identical as well

# Model 2: 1 Intercept and (Separate regressions) one coefficient for each occupation code

# Get one hot encoding of occupation code
X_one_hot = pd.get_dummies(X['Occ_code'])

# Split into train and test
X_train, X_test, y_train, y_test = train_test_split(X_one_hot, y, test_size=0.2,stratify=y, random_state=1234)

# Add constant to X-train for statsmodels
# Adding constant removes affect of other variables
X_train_2 = sm.add_constant(X_train)
X_test_2 = sm.add_constant(X_test)

# building the model and fitting the data
log_reg = sm.Logit(y_train, X_train_2).fit()

# Print Summary
print(log_reg.summary())

# performing predictions on the test datdaset
y_hat = np.array(log_reg.predict(X_test_2))

# Print performance metrics on test set

auroc_sm = roc_auc_score(y_test, y_hat)

if auroc_sm >= 0.5:
    print('sm auroc: ',auroc_sm)
else:
    print('sm auroc: ',(1-auroc_sm))

auprc_sm = average_precision_score(y_test, y_hat)
print('sm auprc: ',auprc_sm)

# Create logistic regression model using scikit learn

logreg_sk = LogisticRegression()
logreg_sk.fit(X_train, y_train.ravel())

logreg_sk.classes_

# Predict probabilities as predicted labels will all be 0
y_pred_sk_p = logreg_sk.predict_proba(X_test)
y_1 = y_pred_sk_p[:,1]
y_0 = y_pred_sk_p[:,0]

logit_roc_auc = roc_auc_score(y_test, y_1)

if logit_roc_auc >= 0.5:
    print('sk auroc_1: ',logit_roc_auc)
else:
    print('sk auroc: ',(1-logit_roc_auc))

auprc = average_precision_score(y_test, y_0)
print('sk auprc: ',auprc)

# Coefficients are very similar when constant added to statsmodels
print('sk coefficients: ',logreg_sk.coef_, logreg_sk.intercept_)



# Model with multiple intercepts?

# Model 3: Stan Hierarchical (Multi-level) Model
# Occupation codes drawn from the same distribution

y_stan = y.reshape(10000,)
# Increase occupation code by 1
X_stan = X.values.reshape(10000,) + 1


# Put data in dictionary format for stan
my_data = {'N':10000,'n_occ':20,'y':y_stan,'X':X_stan}

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
        real <lower=0> tau_occ;
    
        // Parameters in logistic regression
        real intercept; // intercept term in model
        vector[n_occ] occ; //Coefficient for each occupation
       
}

model {
       //hyper priors
       mu_occ ~ normal(0,0.1);
       tau_occ ~ normal(0,1);
       intercept ~ normal(0,0.1);
       
       //priors
       occ ~ normal(mu_occ, tau_occ);
       
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


# Model 4: Pymc3 also with occupation code

# Model 5: Random Forest


 





















