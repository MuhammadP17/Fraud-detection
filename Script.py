import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns



#Import dataset
df = pd.read_csv('Data.zip', compression='zip')
print(df.head())
print(df.describe())


