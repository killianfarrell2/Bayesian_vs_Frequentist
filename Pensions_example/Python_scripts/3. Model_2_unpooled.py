import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score


# Model 2: 1 Intercept and (Separate regressions) one coefficient for each occupation code

sales_data = pd.read_csv('C:\\KF_Repo\\Bayesian_vs_Frequentist\\Pensions_example\\Data\\Pension_sales_train_1.csv') 

X = pd.DataFrame(sales_data['Occupation'].values.reshape(len(sales_data), 1))
y= sales_data['Sale'].values.reshape(len(sales_data), 1)



# Import sales data
sales_data_test = pd.read_csv('C:\\KF_Repo\\Bayesian_vs_Frequentist\\Pensions_example\\Data\\Pension_sales_test_1.csv') 
X_test = pd.DataFrame(sales_data_test['Occupation'].values.reshape(len(sales_data_test), 1))
y_test= sales_data_test['Sale'].values.reshape(len(sales_data_test), 1)

# Get one hot encoding of Occupation
X = pd.get_dummies(X[0])
X_test = pd.get_dummies(X_test[0])

X_train = X
y_train = y

# Convert back to int
X_train  = X_train.astype(str).astype(int)
X_test  = X_test.astype(str).astype(int)

# Split into train and test
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,stratify=y, random_state=1234)

# Add constant to X-train for statsmodels
# Adding constant gives a base probability before occupation taken into account
X_train = sm.add_constant(X_train)
X_test = sm.add_constant(X_test)

# building the model and fitting the data
log_reg = sm.Logit(y_train, X_train).fit()

# Print Summary
print(log_reg.summary())

# performing predictions on the test datdaset
y_hat = np.array(log_reg.predict(X_test))

# Print performance metrics on test set

auroc_sm = roc_auc_score(y_test, y_hat)

if auroc_sm >= 0.5:
    print('sm auroc: ',auroc_sm)
else:
    print('sm auroc: ',(1-auroc_sm))

auprc_sm = average_precision_score(y_test, y_hat)
print('sm auprc: ',auprc_sm)

