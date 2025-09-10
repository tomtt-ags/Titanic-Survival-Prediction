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
# Bivariate analysis involves examining the relationship between two variables.
# For numerical vs numerical we can use scatter plots or correlation matrices
# For categorical vs categorical we can use stacked bar plots or clustered bar plots
# For categorical vs numerical we can use box plots or violin plots or bar plots
# we are going to compare each feature with the target variable "Survived"
print('Bivariate Analysis: Feature vs. Survived')
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
fig.suptitle('Bivariate Analysis with survival', fontsize=16)
# # we will do barplot
sns.barplot(ax = axes[0, 0], x='Pclass', y='Survived', data=titanic_ds).set_title('Survival Rate by Passenger Class')
sns.barplot(ax = axes[0, 1], x='Sex', y='Survived', data=titanic_ds).set_title('Survival Rate by Sex')
sns.barplot(ax = axes[1, 0], x='Embarked', y='Survived', data=titanic_ds).set_title('Survival Rate by Port')
sns.barplot(ax = axes[1, 1], x='Has_Cabin', y='Survived', data=titanic_ds).set_title('Survival Rate by Cabin Availability')
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()
# Interpretation of Categorical Bivariate Analysis:
# Survival Rate by Passenger Class: There is a clear trend showing that passengers in higher classes (
# 1st class) had a significantly higher survival rate compared to those in lower classes (3rd class).
# This suggests that socio-economic status played a crucial role in survival chances.
# Survival Rate by Sex: Female passengers had a significantly higher survival rate compared to male passengers.
# This aligns with the "women and children first" policy that was reportedly followed during the evacuation.
# Survival Rate by Port: Passengers who embarked from Cherbourg (C) had a higher survival rate compared to those from Southampton (S) and Queenstown (Q).
# This could be due to differences in passenger demographics or the timing of the ship's departure from these ports.
# Survival Rate by Cabin Availability: Passengers with cabin information (indicating they had a cabin) had a higher survival rate compared to those without cabin information.
# This might suggest that having a cabin, which is often associated with higher ticket classes, improved survival chances.
# Now we will do bivariate analysis for numerical data
g = sns.FacetGrid(titanic_ds, col='Survived', height=6)
# what this does is create a grid of plots based on the 'Survived' column, so 
# we will get 2 plots one for survived and one for not survived, they are 
# empty rn we have to map a plot to it
g.map(sns.histplot, 'Age', bins=25, kde=True)
plt.suptitle('Age Distribution by Survival Status', y=0.98)
plt.show()
# Interpretation of Numerical Bivariate Analysis:
# Age Distribution by Survival Status: The age distribution of survivors shows a higher concentration of younger passengers
# (especially children and young adults) compared to non-survivors.
# This suggests that younger passengers had a better chance of survival, possibly due to prioritization during evacuation efforts.
# Non-survivors tend to be older, indicating that age was a significant factor in survival chances.
# Overall, younger passengers were more likely to survive the Titanic disaster compared to older passengers.
# Dive into outlier in fare
plt.figure(figsize=(10, 6))
# as we only want 1 plot we aint using subplots
sns.boxplot(y = 'Fare', data=titanic_ds)
plt.title('Box Plot of Fare')
plt.ylabel('Fare')
plt.show()
# Interpretation of Fare Box Plot:
# The box plot of the Fare variable reveals a significant number of outliers, as indicated by
# the points above the upper whisker. The median fare is relatively low, suggesting that most
# passengers paid modest amounts for their tickets. However, the presence of outliers indicates
# that a few passengers paid substantially higher fares, which could be attributed to first-class tickets or
# special accommodations. The interquartile range (IQR) is also quite large, indicating a wide variation in fare prices among passengers.
# Feature Engineering
# Feature engineering is the process of using domain knowledge to create new features or modify existing ones to
# improve the performance of machine learning models.
# We have already created one feature "Has_Cabin"
# Common techniques include:
# - Combining features: Creating new features by combining existing ones, e.g., creating a "FamilySize" feature by summing "SibSp" and "Parch".
# - Extracting from text: e.g. extracting titles from the `Name` column.
# - Binning: COnverting continuous variables into categorical bins, e.g., categorizing `Age` into age groups.
titanic_ds['FamilySize']= titanic_ds['SibSp'] + titanic_ds['Parch'] + 1
# +1 is for the person themselves, so for each person we are counting their siblings/spouses + parents/children + themselves
titanic_ds['IsAlone'] = 0
titanic_ds.loc[titanic_ds['FamilySize'] == 1, 'IsAlone'] = 1
# if family size is 1 then the person is alone, so we set IsAlone
# so the loc line above means:
# Go into the IsAlone column, but only for rows where FamilySize == 1

print("Created 'FamilySize' and 'IsAlone' features:")
print(titanic_ds[['FamilySize', 'IsAlone']].head())
fig, axes = plt.subplots(1, 2, figsize=(12, 6))
sns.barplot(ax = axes[0], x='FamilySize', y='Survived', data=titanic_ds).set_title('Survival Rate by Family Size')
sns.barplot(ax = axes[1], x='IsAlone', y='Survived', data=titanic_ds).set_title('Survival Rate by Is Alone')
plt.show()
# Interpretation of Engineered Features Analysis:
# Survival Rate by Family Size: The survival rate varies with family size. Passengers with a family size of 2-4 had higher survival rates compared to those who were alone or had very
# large families. This suggests that having a moderate family size may have provided better chances of survival, possibly due to mutual support during the evacuation.
# Survival Rate by Is Alone: Passengers who were alone had a significantly lower survival rate compared
# to those who were not alone. This indicates that being part of a group or family may have increased the likelihood of survival, as individuals may have been prioritized for lifeboats or received assistance
# from others during the evacuation process.
# Overall, these engineered features provide valuable insights into the social dynamics that influenced survival chances on the
# Titanic, highlighting the importance of family and companionship during the disaster.
titanic_ds['Title'] = titanic_ds['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
print("Extracted 'Title' from 'Name':")
print(titanic_ds['Title'].value_counts())
titanic_ds['Title'] = titanic_ds['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

titanic_ds['Title'] = titanic_ds['Title'].replace('Mlle', 'Miss')
titanic_ds['Title'] = titanic_ds['Title'].replace('Ms', 'Miss')
titanic_ds['Title'] = titanic_ds['Title'].replace('Mme', 'Mrs')
plt.figure(figsize=(12, 6))
sns.barplot(x='Title', y='Survived', data=titanic_ds)
plt.title('Survival Rate by Title')
plt.ylabel('Survival Probability')
plt.show()
# Interpretation of Title Feature Analysis:
# The survival rates associated with different titles reveal significant variations in survival chances based on social status and gender. 
# For instance, women and children (often referred to as "Ladies" and "Misses") had higher survival rates compared to men ("Masters" and "Mr"). 
# This aligns with the "women and children first" policy that was reportedly followed during the evacuation.
# Additionally, titles such as "Mrs" also show relatively high survival rates, indicating that married women were also prioritized during the evacuation.
# On the other hand, titles associated with higher social status, such as "Sir" and "Lady," also exhibit higher survival rates, suggesting that socio-economic status played a crucial role in survival chances.
# Overall, the analysis of titles provides valuable insights into the social dynamics that influenced survival on the Titanic, highlighting the importance of social class and gender in the face of disaster.
# MULTIVARAITE ANALYSIS
# Multivariate analysis involves examining the relationships between three or more variables simultaneously.
# For numerical vs numerical vs numerical we can use 3D scatter plots or pair plots
# For categorical vs categorical vs categorical we can use mosaic plots or 3D bar plots
# For categorical vs categorical vs numerical we can use clustered bar plots or heatmaps
# For categorical vs numerical vs numerical we can use 3D box plots or violin plots
# For numerical vs numerical vs categorical we can use colored scatter plots or pair plots with hue
sns.catplot(x='Pclass', y='Survived', hue='Sex', data=titanic_ds, kind='bar', height=6, aspect=1.5)
plt.title('Survival Rate by Passenger Class and Sex', y=0.95)
plt.ylabel('Survival Probability')
plt.show()
# Interpretation of Multivariate Analysis
# Survival Rate by Passenger Class and Sex: The analysis reveals that survival rates varied significantly across different passenger classes and between sexes. 
# Female passengers had higher survival rates compared to male passengers within the same class. 
# Additionally, first-class passengers had the highest survival rates, followed by second-class and third-class passengers. 
# This suggests that both gender and socio-economic status played crucial roles in determining survival chances during the Titanic disaster.
plt.figure(figsize=(14, 8))
sns.violinplot(x='Sex', y='Age', hue='Survived', data=titanic_ds, split=True, palette={0: 'blue', 1: 'orange'})
plt.title('Age Distribution by Sex and Survival')
plt.show()
# Interpretation of Age Distribution
# The violin plot illustrates the age distribution of passengers
# based on their sex and survival status. It shows that female passengers, 
# especially those who survived, tended to be younger on average compared to their male counterparts. Additionally, the age
# distribution for male passengers who did not survive appears to be more spread out, indicating a wider range of ages among
# those who perished. This visualization highlights the interplay between age, gender, and survival on the Titanic.
# CORRELATION ANALYSIS
# Correlation analysis is used to measure the strength and direction of the relationship between two numerical variables
plt.figure(figsize=(10, 8))
numeric_cols = titanic_ds.select_dtypes(include=np.number)
correlation_matrix = numeric_cols.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix')
plt.show()
# Interpretation of Correlation Matrix:
# The correlation matrix reveals several key relationships between numerical features in the Titanic dataset.
# Notably, there is a moderate positive correlation between 'Pclass' and 'Fare',
# indicating that passengers in higher classes tended to pay higher fares.
# Additionally, 'Age' shows a slight negative correlation with 'Survived', suggesting that
# younger passengers had a marginally better chance of survival.
# The 'FamilySize' feature, which combines 'SibSp' and 'Parch
# ', shows a weak positive correlation with 'Survived', indicating that passengers with larger families had slightly higher survival rates.
# Overall, while some correlations are evident, most relationships between numerical features are relatively weak, suggesting
# that multiple factors influenced survival on the Titanic.