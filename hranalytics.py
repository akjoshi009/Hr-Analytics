# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 13:49:21 2020

@author: OPO068499
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import lightgbm as lgb
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix


train=pd.read_csv(r"D:\24Projects\HR Analytics\train.csv")
test=pd.read_csv(r"D:\24Projects\HR Analytics\test.csv")

train.isnull().sum()
len(train.columns)

train["department"].value_counts()

le = preprocessing.LabelEncoder()
le.fit(train["department"])
train["department_labeled"]=le.transform(train["department"])
test["department_labeled"]=le.transform(test["department"])

train["region"].value_counts()


train.columns

le = preprocessing.LabelEncoder()
le.fit(train["region"])
train["region_labeled"]=le.transform(train["region"])
test["region_labeled"]=le.transform(test["region"])

train["previous_year_rating"]=train["previous_year_rating"].fillna(4)
test["previous_year_rating"]=test["previous_year_rating"].fillna(4)


train["education"]=train["education"].fillna("Master's & above")
test["education"]=test["education"].fillna("Master's & above")



def educationlevels(txt):
    if(txt=="Bachelor's"):
        return 1
    elif(txt=="Master's & above"):
        return 2
    else:
        return 0
 

train["Education_updated"]=train["education"].apply(educationlevels)
test["Education_updated"]=test["education"].apply(educationlevels)

le = preprocessing.LabelEncoder()
le.fit(train["gender"])
train["gender_labeled"]=le.transform(train["gender"])
test["gender_labeled"]=le.transform(test["gender"])

le = preprocessing.LabelEncoder()
le.fit(train["recruitment_channel"])
train["recruitment_channel_labeled"]=le.transform(train["recruitment_channel"])
test["recruitment_channel_labeled"]=le.transform(test["recruitment_channel"])


def mostpromoted(txt):
    if("Sales" in txt or "Operations" in txt):
        return 3
    elif("Technology"== txt or "Procurement" ==txt or "Analytics" == txt):
        return 2
    else:
        return 1

train["MostImpDept"]=train["department"].apply(mostpromoted)
test["MostImpDept"]=test["department"].apply(mostpromoted)



def getagefar(txt):
    if(txt>34):
       return 1 
    else:
        return 0


def getscorest(txt):
    if(txt>71):
       return 1 
    else:
        return 0

train["avgagefar"]=train["age"].apply(getagefar)
test["avgagefar"]=test["age"].apply(getagefar)

train["avgscorefar"]=train["avg_training_score"].apply(getscorest)
test["avgscorefar"]=test["avg_training_score"].apply(getscorest)


train["Total_Score"]=train["no_of_trainings"]*train["avg_training_score"]
test["Total_Score"]=test["no_of_trainings"]*test["avg_training_score"]

train["ttachivment"] = train['awards_won?']+train['KPIs_met >80%'] + train['previous_year_rating']
test["ttachivment"] = test['awards_won?']+test['KPIs_met >80%'] + test['previous_year_rating']

train["regdept"]=train["region"].astype(str) + train["department"].astype(str)
test["regdept"]=test["region"].astype(str) + test["department"].astype(str)


train["jonag"]=train["age"]-train["length_of_service"]
test["jonag"]=test["age"]-test["length_of_service"]
  


le.fit(train["regdept"])
train["regdept"]=le.transform(train["regdept"])
test["regdept"]=le.fit_transform(test["regdept"])



bins = [50,60,70,80,90]
labels = [1,2,3,4]
train['binnedage'] = pd.cut(train['age'], bins=bins, labels=labels)
test['binnedage'] = pd.cut(test['age'], bins=bins, labels=labels)


dd=train[train["is_promoted"]==1]["avg_training_score"]


test["department"].value_counts()

dtrain=train[train["length_of_service"]<30]
dtrain=dtrain[dtrain["length_of_service"]>1]
dtrain=dtrain[dtrain["age"]<59]

#dtrain["avg_training_score"]=np.log(train["avg_training_score"])
#test["avg_training_score"]=np.log(test["avg_training_score"])
dtrain.columns


colx=[ 'region_labeled', 
       'no_of_trainings', 'previous_year_rating',
       'length_of_service', 'KPIs_met >80%', 'awards_won?',
       'avg_training_score', 'department_labeled','binnedage',
       'region_labeled', 'Education_updated', 'gender_labeled',
       'MostImpDept','Total_Score', 'ttachivment', 'regdept', 'jonag'
      ]


train["education"].value_counts()

X_train,X_test, y_train, y_test = train_test_split(
dtrain[colx],dtrain['is_promoted'], test_size=0.33, random_state=42)


clf=lgb.LGBMClassifier(boosting_type='gbdt',class_weight=None, colsample_bytree=1.0,
               importance_type='split', learning_rate=0.1, max_depth=-1,
               min_child_samples=20, min_child_weight=0.001, min_split_gain=0.0,
               n_estimators=500, n_jobs=-1, num_leaves=31, objective=None,
               random_state=None, reg_alpha=0.0, reg_lambda=0.0, silent=True,
               subsample=1.0, subsample_for_bin=200000, subsample_freq=0)

clf.fit(X_train,y_train)
pred=clf.predict(X_test)
accuracy_score(y_test, pred)
confusion_matrix(y_test, pred)


feat_importances = pd.Series(clf.feature_importances_, index=dtrain[colx].columns)
feat_importances.plot(kind='barh')


clf.fit(dtrain[colx],dtrain['is_promoted'])
pred=clf.predict(test[colx])

test["is_promoted"]=pred
test["is_promoted"].value_counts()

test[["employee_id","is_promoted"]].to_csv(r"D:\24Projects\HR Analytics\output.csv",index=False)




