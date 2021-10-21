import pandas as pd
import numpy as np

# Function to Create Sales data for pensions based on Occupation
def create_sales_data(total,targets,profession='Doctor'):
    
    # Create list of targets with 10% Prior
    targets = np.concatenate((np.zeros(total-targets),np.ones(targets))).reshape((total, 1))
    # Convert to int
    targets = targets.astype('int32')

    # Create Dataframe
    data = pd.DataFrame(targets)
    data.columns = ['Sale']
    data['Occupation'] = profession
    return data

# Create dataframes for each profession for training
doctor = create_sales_data(80,4,'doctor')
solicitor = create_sales_data(320,8,'solicitor')
engineer = create_sales_data(240,4,'engineer')
sole_trader = create_sales_data(240,40,'sole_trader')
banker = create_sales_data(320,16,'banker')
nurse = create_sales_data(800,16,'nurse')
hgv_driver = create_sales_data(640,160,'hgv_driver')
retail_sales = create_sales_data(720,120,'retail_sales')
software_dev = create_sales_data(240,8,'software_dev')
accountant = create_sales_data(400,4,'accountant')

# Combine dataframes
frames = [doctor, solicitor, engineer,sole_trader,banker,nurse,hgv_driver,retail_sales,software_dev,accountant]
# Create dataframe
result = pd.concat(frames)

# Save file as csv
file_name = 'C:\\KF_Repo\\Bayesian_vs_Frequentist\\Pensions_example\\Data\\Pension_sales_train_1.csv'
result.to_csv(file_name,index=False)

# Create test set (20%)

doctor = create_sales_data(20,1,'doctor')
solicitor = create_sales_data(80,2,'solicitor')
engineer = create_sales_data(60,1,'engineer')
sole_trader = create_sales_data(60,10,'sole_trader')
banker = create_sales_data(80,4,'banker')
nurse = create_sales_data(200,4,'nurse')
hgv_driver = create_sales_data(160,40,'hgv_driver')
retail_sales = create_sales_data(180,30,'retail_sales')
software_dev = create_sales_data(60,2,'software_dev')
accountant = create_sales_data(100,1,'accountant')

# Combine dataframes
frames = [doctor, solicitor, engineer,sole_trader,banker,nurse,hgv_driver,retail_sales,software_dev,accountant]
# Create dataframe
result = pd.concat(frames)

# Save file as csv
file_name = 'C:\\KF_Repo\\Bayesian_vs_Frequentist\\Pensions_example\\Data\\Pension_sales_test_1.csv'
result.to_csv(file_name,index=False)


# Create reduced sized datasets with the same proportions


# Create dataframes for each profession for training
doctor = create_sales_data(20,1,'doctor')
solicitor = create_sales_data(80,2,'solicitor')
engineer = create_sales_data(60,1,'engineer')
sole_trader = create_sales_data(60,10,'sole_trader')
banker = create_sales_data(80,4,'banker')
nurse = create_sales_data(200,4,'nurse')
hgv_driver = create_sales_data(160,40,'hgv_driver')
retail_sales = create_sales_data(180,30,'retail_sales')
software_dev = create_sales_data(75,2,'software_dev')
accountant = create_sales_data(100,1,'accountant')

# Combine dataframes
frames = [doctor, solicitor, engineer,sole_trader,banker,nurse,hgv_driver,retail_sales,software_dev,accountant]
# Create dataframe
result = pd.concat(frames)

# Save file as csv
file_name = 'C:\\KF_Repo\\Bayesian_vs_Frequentist\\Pensions_example\\Data\\Pension_sales_train_2.csv'
result.to_csv(file_name,index=False)

# Create test set (20%)

doctor = create_sales_data(5,1,'doctor')
solicitor = create_sales_data(20,1,'solicitor')
engineer = create_sales_data(15,1,'engineer')
sole_trader = create_sales_data(15,2,'sole_trader')
banker = create_sales_data(20,1,'banker')
nurse = create_sales_data(50,1,'nurse')
hgv_driver = create_sales_data(40,10,'hgv_driver')
retail_sales = create_sales_data(45,8,'retail_sales')
software_dev = create_sales_data(15,1,'software_dev')
accountant = create_sales_data(25,1,'accountant')

# Combine dataframes
frames = [doctor, solicitor, engineer,sole_trader,banker,nurse,hgv_driver,retail_sales,software_dev,accountant]
# Create dataframe
result = pd.concat(frames)

# Save file as csv
file_name = 'C:\\KF_Repo\\Bayesian_vs_Frequentist\\Pensions_example\\Data\\Pension_sales_test_2.csv'
result.to_csv(file_name,index=False)













