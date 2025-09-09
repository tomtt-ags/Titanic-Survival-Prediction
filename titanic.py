import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
pd.set_option("display.max_rows", None)  
pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)  
pd.set_option("display.colheader_justify", "left")
sns.set(style="whitegrid")
titanic_ds = pd.read_csv('Titanic_Survival_Dataset/Datasets/Titanic-Dataset.csv')
# print("First 5 rows of the Titanic dataset:")
# this gets us the first 5 rows of the dataset
print(titanic_ds.head())
# lets get a summary of the dataset
print(titanic_ds.info())
# Interpretation of `.info()`:
# - The dataset contains 891 entries (passengers) and 12 columns.
# - Missing Values Identified:** `Age`, `Cabin`, and `Embarked` have missing values. 
# `Cabin` is missing a significant amount of data (~77%), which will require special attention.

# lets get some basic statistics of the dataset
print(titanic_ds.describe())
# Interpretation of `.describe()`:
# Survived: About 38.4% of passengers in this dataset survived.
# Age: The age ranges from ~5 months to 80 years, with an average age of about 30.
# Fare: The fare is highly skewed, with a mean of $32 but a median of only $14.45.
# The maximum fare is over $512, indicating the presence of extreme outliers.
# print(titanic_ds['Cabin'].value_counts())
print(titanic_ds['Cabin'].count())
# Time to clean data
print(titanic_ds.isna().sum())
# Handling Missing Values
# numerical data - Age we will use median to avoid outliers
# categorical data - Embarked we will use mode
# data like cabin which have many missing values we can either drop or 
# re-engineer the data to create a new feature like "HasCabin" (Yes/No)
median_age = titanic_ds['Age'].median()
titanic_ds['Age'] = titanic_ds['Age'].fillna(median_age)  
# now we have fixed age missing values
# print(titanic_ds[['Age', 'Embarked', 'Cabin']].isna().sum())
mode_embarked = titanic_ds['Embarked'].mode()[0]
titanic_ds['Embarked'] = titanic_ds['Embarked'].fillna(mode_embarked)
# now we have fixed Embarked missing values
# print(titanic_ds[['Age', 'Embarked', 'Cabin']].isna().sum())
# now we will create a new feature "HasCabin"
titanic_ds['Has_Cabin'] = titanic_ds['Cabin'].notna().astype(int)
# now we can drop the Cabin column
titanic_ds.drop(columns=['Cabin'], inplace = True)
print(titanic_ds.isna().sum())
# now we have fixed all missing values
# UNIVARIATE ANALYSIS 
# simplest form of data analysis, where we examine the distribution of a single variable.
# so for numerical values we can use histograms or kernel density plots for distribution
# for numerical we also use box plots to identify outliers, central tendency and spread
# for categorical we can use bar plots or pie charts to see the frequency/count
# or proportion of each category
fig, axes = plt.subplots(2, 3, figsize=(12, 8))
fig.suptitle('Univariate Analysis of Categorical Features', fontsize=16)
# fig and axes are the figure and axes objects(2 different val returned by subplots function)  

# seaborn.countplot() is a function in the Seaborn library in Python 
# used to display the counts of observations in categorical data. 
# It shows the distribution of a single categorical variable or 
# the relationship between two categorical variables by creating a bar plot.
# as its univartate we won't specify y parameter and 
# y axis will be the count of each category by default
check = titanic_ds.columns
print(check)
sns.countplot(ax=axes[0, 0], x='Survived', data=titanic_ds).set_title('Survival Distribution')
sns.countplot(ax=axes[0, 1], x='Pclass', data=titanic_ds).set_title('Passenger Class Distribution')
sns.countplot(ax=axes[0, 2], x='Sex', data=titanic_ds).set_title('Gender Distribution')
sns.countplot(ax=axes[1, 0], x='Embarked', data=titanic_ds).set_title('Port of Embarkation')
sns.countplot(ax=axes[1, 1], x='SibSp', data=titanic_ds).set_title('Siblings/Spouses Aboard')
sns.countplot(ax=axes[1, 2], x='Parch', data=titanic_ds).set_title('Parents/Children Aboard')
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()
# Interpretation of Categorical Univariate Analysis:
# Survival Distribution: The dataset shows a significant imbalance between survivors and non-survivors, with
# a larger proportion of passengers not surviving.
# Passenger Class Distribution: The majority of passengers were in the 3rd class, followed by 1st and 2
# nd classes.
# Gender Distribution: The dataset has more female passengers than male passengers.
# Port of Embarkation: Most passengers boarded the Titanic at Southampton (S), followed by Cherbourg (C) and Queenstown (Q).
# Siblings/Spouses Aboard: The majority of passengers had no siblings or spouses aboard.
# Parents/Children Aboard: Most passengers had no parents or children aboard.
# Now we will do univariate analysis for numerical data
fig, axes = plt.subplots(1, 2, figsize=(12, 8))
fig.suptitle('Univariate Analysis of Numerical Features', fontsize=16)
sns.histplot(ax=axes[0], data=titanic_ds, x='Age', kde=True, bins=30).set_title('Age Distribution')
# kde=True adds a kernel density estimate line to the histogram
# bins is basically the number of bars in the histogram
sns.histplot(ax=axes[1], data=titanic_ds, x='Fare', kde=True,bins=30).set_title('Fare Distribution')
plt.show()
# Interpretation of Numerical Univariate Analysis:
# Age Distribution: The age distribution of passengers is right-skewed, with a peak around 20-30 years. 
# There are fewer older passengers, and the distribution suggests that most passengers were relatively young.
# Fare Distribution: The fare distribution is highly right-skewed, with a long tail extending towards higher fare values. 
# Most passengers paid lower fares, with a significant  number paying below $50. The presence of outliers is evident, with some passengers paying very high fares.
# BIVARIATE ANALYSIS