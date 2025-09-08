import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="whitegrid")
titanic_ds = pd.read_csv('Titanic_Survival_Dataset/Datasets/Titanic-Dataset.csv')
print(titanic_ds.head())