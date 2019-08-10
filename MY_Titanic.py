# -*- coding: utf-8 -*-
"""
Created on Wed Aug  7 23:59:06 2019

@author: Owais
"""
#importing some libraries
import pandas as pd
import numpy as np
import random as rnd
from collections import Counter
#importing libraries for data visualization and ignoring warnings

import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

import warnings
warnings.filterwarnings('ignore') 

#importing dataset
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
ids=test['PassengerId']

#take a look at the training data
train.describe(include="all")

#get a list of the features within the dataset
print(train.columns)

#see a sample of the dataset to get an idea of the variables
train.sample(5)

#check for any other unusable values
print(pd.isnull(train).sum())


'''# Outlier detection 
def detect_outliers(df,n,features):
    """
    Takes a dataframe df of features and returns a list of the indices
    corresponding to the observations containing more than n outliers according
    to the Tukey method.
    """
    outlier_indices = []
    
    # iterate over features(columns)
    for col in features:
        # 1st quartile (25%)
        Q1 = np.percentile(df[col], 25)
        # 3rd quartile (75%)
        Q3 = np.percentile(df[col],75)
        # Interquartile range (IQR)
        IQR = Q3 - Q1
        
        # outlier step
        outlier_step = 1.5 * IQR
        
        # Determine a list of indices of outliers for feature col
        outlier_list_col = df[(df[col] < Q1 - outlier_step) | (df[col] > Q3 + outlier_step )].index
        
        
         # append the found outlier indices for col to the list of outlier indices 
        outlier_indices.extend(outlier_list_col)
        
    # select observations containing more than 2 outliers
    outlier_indices = Counter(outlier_indices)        
    multiple_outliers = list( k for k, v in outlier_indices.items() if v > n )
    
    return multiple_outliers   

# detect outliers from Age, SibSp , Parch and Fare
Outliers_to_drop = detect_outliers(train,2,["Age","SibSp","Parch",'Fare'])

#list of rows or outliers to drop
train.loc[Outliers_to_drop]

# Drop outliers
train = train.drop(Outliers_to_drop, axis = 0).reset_index(drop=True)'''

#Data visualization
#feature Sex
sns.barplot(x="Sex", y="Survived", data=train)
train[["Sex","Survived"]].groupby('Sex').mean()
from sklearn.preprocessing import LabelEncoder
label=LabelEncoder()
train['Sex']=label.fit_transform(train['Sex'])
test['Sex']=label.fit_transform(test['Sex'])

#feature Pclass
sns.barplot(x="Pclass", y="Survived", data=train)
train[["Pclass","Survived"]].groupby('Pclass').mean()

#feature SibSP
sns.barplot(x="SibSp", y="Survived", data=train)
train[["SibSp","Survived"]].groupby('SibSp').mean()

#feature Parch
sns.barplot(x="Parch", y="Survived", data=train)
train[["Parch","Survived"]].groupby('Parch').mean()

#feature Age
train["Age"].isnull().sum()
train["Age"] = train["Age"].fillna(train["Age"].median())
test["Age"] = test["Age"].fillna(test["Age"].median())
g=sns.FacetGrid(train, col='Survived')
g.map(sns.distplot,'Age')

#feature cabin
train["CabinBool"] = (train["Cabin"].notnull().astype('int'))
test["CabinBool"] = (test["Cabin"].notnull().astype('int'))
sns.barplot(x="CabinBool", y="Survived", data=train)
train=train.drop(['Cabin'],axis=1)
test=test.drop(['Cabin'],axis=1)
#feature Embarked
#replacing missing value by  more frequent
train["Embarked"].mode()
train["Embarked"] = train["Embarked"].fillna("S")
# Explore Embarked vs Survived 
g = sns.factorplot(x="Embarked", y="Survived",  data=train,
                   size=6, kind="bar", palette="muted")
g.despine(left=True)
g = g.set_ylabels("survival probability")

embarked_dummies=pd.get_dummies(train.Embarked, prefix='Embarked')
train=pd.concat([train,embarked_dummies],axis=1)

embarked_dummies_test=pd.get_dummies(test.Embarked,prefix='Embarked')
test=pd.concat([test,embarked_dummies_test],axis=1)
train=train.drop(['Embarked'],axis=1)
test=test.drop(['Embarked'],axis=1)

#feature fare
train["Fare"].isnull().sum()
train[["Fare","Survived"]].groupby('Fare').mean()
test.isnull().sum()
test["Fare"] = test["Fare"].fillna(test["Fare"].median())

#Y-train ie survived
Y_train=train['Survived']

#droping useless columns
test=test.drop(['PassengerId','Ticket'],axis=1)
train=train.drop(['PassengerId','Ticket'],axis=1)




#combining dataset for feature engineering in Name
train_len = len(train)
dataset =  pd.concat(objs=[train, test], axis=0).reset_index(drop=True)

#feature Name
dataset['Name'].head()

# Get Title from Name
dataset['Title'] = dataset['Name'].str.split(", ", expand=True)[1].str.split(".", expand=True)[0]
dataset["Title"].head()
pd.crosstab(dataset['Title'], dataset['Sex'])

g = sns.countplot(x="Title",data=dataset)
g = plt.setp(g.get_xticklabels(), rotation=45)

# Convert to categorical values Title 
dataset["Title"] = dataset["Title"].replace(['Lady', 'the Countess','Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
dataset["Title"] = dataset["Title"].map({"Master":0, "Miss":1, "Ms" : 1 , "Mme":1, "Mlle":1, "Mrs":1, "Mr":2, "Rare":3})
#countplot
g = sns.countplot(dataset["Title"])
g = g.set_xticklabels(["Master","Miss/Ms/Mme/Mlle/Mrs","Mr","Rare"])
#barplot
g=sns.barplot(x="Title", y="Survived", data=dataset)
g = g.set_xticklabels(["Master","Miss/Ms/Mlle/Mme/Mrs","Mr","Rare"])

#dropping useless column
dataset=dataset.drop(['Name','Survived'],axis=1)
#train=train.drop(['Survived'],axis=1)

#let separate training and test set 
train = dataset[:train_len]
test = dataset[train_len:]
#test=test.drop(["Survived"],axis = 1)

#Fit Our different Models and see accuracy
#importing libraries again
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score

#LogisticRegression
lgr=LogisticRegression(random_state = 0)
lgr.fit(train, Y_train)
y_pred = lgr.predict(test)
Y_pred_acc=lgr.predict(train)
acc_lgr = round(accuracy_score(Y_pred_acc, Y_train)*100,2)
print(acc_lgr)

#SVC
svc=SVC()
svc.fit(train, Y_train)
y_pred = svc.predict(test)
Y_pred_acc=svc.predict(train)
acc_svc = round(accuracy_score(Y_pred_acc, Y_train)*100,2)
print(acc_svc)
#LinearSVC
L_svc=LinearSVC()
L_svc.fit(train, Y_train)
y_pred = L_svc.predict(test)
Y_pred_acc=L_svc.predict(train)
acc_L_svc = round(accuracy_score(Y_pred_acc, Y_train)*100,2)
print(acc_L_svc)
#RandomForestClassifier
rfc=RandomForestClassifier(n_estimators=51,criterion='entropy',random_state=0)
rfc.fit(train, Y_train)
y_pred = rfc.predict(test)
Y_pred_acc=rfc.predict(train)
acc_rfc = round(accuracy_score(Y_pred_acc, Y_train)*100,2)
print(acc_rfc)
#KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=3,weights='distance',algorithm='auto',p=1)
knn.fit(train, Y_train)
y_pred = knn.predict(test)
Y_pred_acc=knn.predict(train)
acc_knn = round(accuracy_score(Y_pred_acc, Y_train)*100,2)
print(acc_knn)
#GaussianNB
gNB=GaussianNB()
gNB.fit(train, Y_train)
y_pred = gNB.predict(test)
Y_pred_acc=gNB.predict(train)
acc_gNB = round(accuracy_score(Y_pred_acc, Y_train)*100,2)
print(acc_gNB)
#Perceptron
pcn=Perceptron()
pcn.fit(train, Y_train)
y_pred = pcn.predict(test)
Y_pred_acc=pcn.predict(train)
acc_pcn = round(accuracy_score(Y_pred_acc, Y_train)*100,2)
print(acc_pcn)
#SGDClassifier
sdg=SGDClassifier()
sdg.fit(train, Y_train)
y_pred = sdg.predict(test)
Y_pred_acc=sdg.predict(train)
acc_sdg = round(accuracy_score(Y_pred_acc, Y_train)*100,2)
print(acc_sdg)
#DecisionTreeClassifier
DTC=DecisionTreeClassifier(criterion='entropy',class_weight='balanced')
DTC.fit(train, Y_train)
y_pred = DTC.predict(test)
Y_pred_acc=DTC.predict(train)
acc_DTC = round(accuracy_score(Y_pred_acc, Y_train)*100,2)
print(acc_DTC)
#gradient booster
gbk = GradientBoostingClassifier(loss='deviance',learning_rate=0.1,n_estimators=50,)
gbk.fit(train, Y_train)
y_pred = gbk.predict(test)
acc_gbk = round(accuracy_score(Y_pred_acc, Y_train) * 100, 2)
print(acc_gbk)
#comparision of  accuracies
Classifiers = pd.DataFrame({
    'Classifier': ['Support Vector Machine', 'KNN', 'Logistic Regression', 
              'Random Forest', 'Naive Bayes', 'Perceptron', 'Linear SVC', 
              'Decision Tree', 'Stochastic Gradient Descent', 'Gradient Boosting Classifier'],
    'Accuracy': [acc_svc, acc_knn, acc_lgr, 
              acc_rfc, acc_gNB, acc_pcn,acc_L_svc, acc_DTC,
              acc_sdg, acc_gbk]})
Classifiers.sort_values(by='Accuracy', ascending=False)

#Predict of survival for test set using KNN
Predictions = knn.predict(test)

#set the output as a dataframe and convert to csv file named submission.csv
output = pd.DataFrame({ 'PassengerId' : ids, 'Survived': Predictions })
output.to_csv('submission.csv', index=False)