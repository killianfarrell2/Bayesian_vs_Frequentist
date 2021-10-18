import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
import pystan
import arviz as az


# Model 5: Stan One hot encoding model
# Import Sales data
sales_data = pd.read_csv('C:\\KF_Repo\\Bayesian_vs_Frequentist\\Pensions_example\\Data\\Pension_sales.csv') 

X = pd.DataFrame(sales_data['Occupation'].values.reshape(len(sales_data), 1))
y= sales_data['Sale'].values.reshape(len(sales_data), 1)


# Get one hot encoding of Occupation
X = pd.get_dummies(X[0])
# Convert to integers
X = X.astype(str).astype(int)

# Split into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,stratify=y, random_state=1234)

# Reshape arrays so they can be taken by Stan
y_train = y_train.reshape(len(y_train),)
X_train = X_train.values.reshape(len(X_train),10)


# Put data in dictionary format for stan
my_data = {'N':len(X_train),'n_occ':len(np.unique(X_train)),'y':y_train,'X':X_train }

my_code = """
data {
      int N; // number of data points
      int<lower=1> K; // Number of categories
      matrix[N,K]  X;
      vector [N] y;
    
}

parameters {
    
          vector[K] beta;
  real<lower=0> sigma;
    
        // Parameters in logistic regression
        real intercept; // intercept term in model
        real mu_x0; // mean x0
        real <lower=0> sd_x0; //sd x0
        real mu_x1; // mean x1
        real <lower=0> sd_x1; //sd x1
        
        real  coef_x0;
        real  coef_x1;
       
}

model { 
       mu_x0 ~ normal(0,3);
       sd_x0 ~ normal(0,3);
       mu_x1 ~ normal(0,3);
       sd_x1 ~ normal(0,3);
       intercept ~ normal(0,3);
       coef_x0 ~ normal(mu_x0, sd_x0);
       coef_x1 ~ normal(mu_x1, sd_x1);
       y ~ bernoulli_logit(intercept + coef_x0 + coef_x1);   
       
       }

"""


  //likelihood
       y ~ bernoulli_logit(intercept + coef_x0* + coef_x1);
transformed parameters {
  real theta1; //theta for equation

  theta1 = intercept + coef_x0* + coef_x1*
  
  theta1 = exp(home + att[ht] - def[at]);

}


#generated quantities {
#    vector [n_occ] prob;
#    for (n in 1:n_occ) 
#    prob[n] = inv_logit(intercept + occ[n]); 
#}

# Create Model - this will help with recompilation issues
stan_model = pystan.StanModel(model_code=my_code)

# Call sampling function with data as argument
fit = stan_model.sampling(data=my_data, iter=2000, chains=4, seed=1,warmup=1000)

# Put Posterior draws into a dictionary
params = fit.extract()

# Extract generated samples
prob = fit.extract()['prob']

# Get summary statistics for parameters
print(fit)


detailed_summary = fit.summary()


# visual summary
fit.plot()

