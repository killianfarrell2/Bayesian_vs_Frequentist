# Create list of targets with 10% Prior
targets = np.concatenate((np.zeros(9000),np.ones(1000))).reshape((10000, 1))
random.shuffle(targets)

# Generate a Normally distributed sample of ca balances
# mean is 1000 and sd is 500
ca_balance = np.random.normal(2000, 1000, 10000).reshape((10000, 1))
random.shuffle(ca_balance)

# Combine columns
data = pd.DataFrame(np.concatenate((ca_balance,targets),axis=1))
data.columns = ['ca_balance','sale']

