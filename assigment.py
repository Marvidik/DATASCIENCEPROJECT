import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline


url = 'https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv'
titanic_df = pd.read_csv(url)

titanic_df.head()
titanic_df.info()
titanic_df.isnull().sum()
titanic_df['Age'].fillna(titanic_df['Age'].median(), inplace=True)
titanic_df.dropna(subset=['Cabin', 'Embarked'], inplace=True)
titanic_df['Sex'] = pd.Categorical(titanic_df['Sex'])
titanic_df['Embarked'] = pd.Categorical(titanic_df['Embarked'])
titanic_df['Family_Size'] = titanic_df['SibSp'] + titanic_df['Parch']
sns.countplot(x='Survived', data=titanic_df)


sns.histplot(x='Age', data=titanic_df)

sns.countplot(x='Survived', hue='Sex', data=titanic_df)
sns.countplot(x='Survived', hue='Pclass', data=titanic_df)
sns.countplot(x='Survived', hue='Embarked', data=titanic_df)
sns.countplot(x='Survived', hue='Family_Size', data=titanic_df)

titanic_df.describe()