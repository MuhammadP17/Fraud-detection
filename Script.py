import os
from collections import Counter

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
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification


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


#Now split original dataset
X = df.drop(['Class'], axis=1)
y = df['Class']

#DO NOT SCALE VARIABLES FIRST, THIS CAUSES DATA LEAKING.
#Split the dataset

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y, shuffle=True)

X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=0, stratify=y_temp, shuffle=True)


#Now we can scale and transform the variables.

scalar = RobustScaler()

scalar.fit(X_train)

X_train = scalar.transform(X_train)
X_val = scalar.transform(X_val)
X_test = scalar.transform(X_test)

#How do we replace the columns??

#Create the balanced data sample






#Then need to split the data into a training and testing set respectively.

#Correlation matrix with subsample to check for relationships between variables.

#And some more stuff.


