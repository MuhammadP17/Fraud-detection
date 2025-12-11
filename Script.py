import os
import pandas as pd
import numpy as np
import matplotlib
if 'PYCHARM_HOSTED' in os.environ:
    matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold


#Makes sure new data is generated for testing purposes in each run.
np.random.seed(0)


#Import dataset and check properties.
df = pd.read_csv('Data.zip', compression='zip')
print(df.head())
print(df.describe())
print(df.info())

#Check for null values
print(df.isnull().sum().max())


#Check dataset for imbalance
sns.countplot(x='Class', data=df, palette='Set3')
plt.title("Count by Class (Fraud = 1)")

#Check the distributions of each variable.
df.hist(bins=20)
plt.show()


#Scale variables for better model performance and accuracy.
scalar = RobustScaler()
df['r_amount'] = scalar.fit_transform(df[['Amount']])
df['r_time'] = scalar.fit_transform(df[['Time']])

df.drop(['Amount', 'Time'], axis=1, inplace=True)


#Now split original dataset
X = df.drop(['Class'], axis=1)
y = df['Class']






#Also need to remove outliers for better model efficiency.

#Then need to split the data into a training and testing set respectively.

#Correlation matrix with subsample to check for relationships between variables.

#And some more stuff.


