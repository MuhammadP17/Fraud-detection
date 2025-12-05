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
plt.show()


#Check for and remove outliers.


#Need to make the dataset balanced to make an accurate enough model.


#Also need to remove outliers for better model efficiency.

#Then need to split the data into a training and testing set respectively.

#Correlation matrix with subsample to check for relationships between variables.

#And some more stuff.


