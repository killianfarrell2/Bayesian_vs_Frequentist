import pystan
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random

# Create list of targets with 10% Prior
targets = np.concatenate((np.zeros(9000),np.ones(1000)))
random.shuffle(targets)
# reshape after shuffle
targets = targets.reshape(10000, 1)


# Generate a Normally distributed sample of ca balances
# mean is 1000 and sd is 500
ca_balance = np.random.normal(2000, 1000, 10000).reshape((10000, 1))

# Create categorical data (occupation codes) 20 in total
occupation_codes = np.random.randint(20, size=(10000)).reshape((10000, 1))

# Combine columns
data = pd.DataFrame(np.concatenate((ca_balance,occupation_codes,targets),axis=1))
data.columns = ['ca_balance','Occ_code','sale']

y = data[['sale']]
X = data[['ca_balance','Occ_code']]


from sklearn.model_selection import train_test_split


# When using stratified split it messes up AUROC
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

import statsmodels.api as sm
# Add constant to X-train for statsmodels
# Adding constant removes affect of other variables
X_train_2 = sm.add_constant(X_train)
X_test_2 = sm.add_constant(X_test)

# building the model and fitting the data
log_reg = sm.Logit(y_train, X_train_2).fit()

# Create Summary
sum_1 = log_reg.summary()

print(sum_1)

# performing predictions on the test datdaset
y_hat = np.array(log_reg.predict(X_test_2))

# Print performance metrics on test set
from sklearn.metrics import roc_auc_score
auroc_sm = roc_auc_score(y_test.values, y_hat)
print(auroc_sm)

from sklearn.metrics import average_precision_score
auprc_sm = average_precision_score(y_test.values, y_hat)
print(auprc_sm)

# Create logistic regression model using scikit learn
from sklearn.linear_model import LogisticRegression
logreg_sk = LogisticRegression()
logreg_sk.fit(X_train, y_train.values.ravel())

# Predict probabilities as predicted labels will all be 0
y_pred_sk_p = logreg_sk.predict_proba(X_test)
y_1 = y_pred_sk_p[:,1]

logit_roc_auc = roc_auc_score(y_test, y_1)
print(logit_roc_auc)

auprc = average_precision_score(y_test.values, y_1)
print(auprc)

# Coefficients are very similar when constant added to statsmodels
print(logreg_sk.coef_, logreg_sk.intercept_)

# Results are almost identical as well




 





















