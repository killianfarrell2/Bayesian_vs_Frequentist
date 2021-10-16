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

# Create dataframes for each profession
doctor = create_sales_data(100,5,'doctor')
solicitor = create_sales_data(400,10,'solicitor')
engineer = create_sales_data(300,5,'engineer')
sole_trader = create_sales_data(300,50,'sole_trader')
banker = create_sales_data(400,20,'banker')
nurse = create_sales_data(1000,20,'nurse')
hgv_driver = create_sales_data(800,200,'hgv_driver')
retail_sales = create_sales_data(900,150,'retail_sales')
software_dev = create_sales_data(300,10,'software_dev')
accountant = create_sales_data(500,5,'accountant')

# Combine dataframes
frames = [doctor, solicitor, engineer,sole_trader,banker,nurse,hgv_driver,retail_sales,software_dev,accountant]
# Create dataframe
result = pd.concat(frames)

# Save file as csv
file_name = 'C:\\KF_Repo\\Bayesian_vs_Frequentist\\Pensions_example\\Data\\Pension_sales.csv'

result.to_csv(file_name,index=False)




