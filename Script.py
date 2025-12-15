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
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
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


#Check how much the total of the fraud amounts are relative to the sum of all the transactions.
total = df['Amount'].sum()
subtotal = df.loc[df['Class'] == 1]['Amount'].sum()
prop = subtotal / total * 100
print("This is the proportion of fraud based on amount:", round(prop, 2), "%")


#Check the distributions of each variable.
df.hist(bins=20)
plt.show()


#Split the dataset
X = df.drop(['Class'], axis=1)
y = df['Class']
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y, shuffle=True)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=0, stratify=y_temp, shuffle=True)


#Now we can scale and transform the variables.
cols = X_train.columns
scalar = RobustScaler()
scalar.fit(X_train)
X_train = scalar.transform(X_train)
X_val = scalar.transform(X_val)
X_test = scalar.transform(X_test)


#Create the fresh datasets
X_train = pd.DataFrame(X_train, columns=cols)
X_val = pd.DataFrame(X_val, columns=cols)
X_test = pd.DataFrame(X_test, columns=cols)


#Create the balanced data sample(To be used for training your models)
print('Initial training dataset shape %s' % Counter(y))
rus = RandomUnderSampler(random_state=0, sampling_strategy='majority', replacement=False)
X_res, y_res = rus.fit_resample(X_train, y_train)
print('Training dataset shape %s' % Counter(y_res))
X_res = pd.DataFrame(X_res, columns=cols)


#Check correlations between variables
original_corr = df.corr()
sns.heatmap(original_corr, cmap='coolwarm_r')
plt.title("Correlation Heatmap from original dataset")
plt.show()

sub_corr = X_res.corr()
sns.heatmap(sub_corr, cmap='coolwarm_r')
plt.title("Correlation Heatmap from subsample")
plt.show()


#Create the benchmark to beat.
print("Logistic regression benchmarks: ")
model = LogisticRegression()
model.fit(X_res, y_res)
y_pred = model.predict(X_val)

print(confusion_matrix(y_val, y_pred))
print(classification_report(y_val, y_pred))


#Export all datasets to csv for more machine learning
X_train.to_csv('X_train.csv', index=False)
X_val.to_csv('X_val.csv', index=False)
X_test.to_csv('X_test.csv', index=False)
y_train.to_csv('y_train.csv', index=False)
y_val.to_csv('y_val.csv', index=False)
y_test.to_csv('y_test.csv', index=False)
X_res.to_csv('X_train_balanced.csv', index=False)
y_res.to_csv('y_train_balanced.csv', index=False)








