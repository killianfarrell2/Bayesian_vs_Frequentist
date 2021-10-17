import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score


# Model 1: Intercept and one coefficient for all occupation codes

# Import Sales data
sales_data = pd.read_csv('C:\\KF_Repo\\Bayesian_vs_Frequentist\\Pensions_example\\Data\\Pension_sales.csv') 

X = pd.DataFrame(sales_data['Occupation'].values.reshape(len(sales_data), 1))

y= sales_data['Sale'].values.reshape(len(sales_data), 1)

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

