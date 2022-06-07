import numpy as np 
import pandas as pd
DS=pd.read_csv("HousingData.csv")
DS.head()
DS.describe()
DS.isnull().sum()
DS.info()
DS[['CRIM','ZN','INDUS','AGE','LSTAT']] = DS[['CRIM','ZN','INDUS','AGE','LSTAT']].fillna((DS[['CRIM','ZN','INDUS','AGE','LSTAT']].mean()))
DS['CHAS']=DS['CHAS'].fillna(method='bfill')
X=DS.iloc[:,0:-1]
Y=DS.iloc[:,-1]
X.head()
Y.head()
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=0)
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(max_depth=6)
DT_reg=regressor.fit(X_train, Y_train)
DT_reg
print(DT_reg.score(X_train,Y_train))
print(DT_reg.score(X_test,Y_test))
Y_pred=DT_reg.predict(X_test)
Y_pred
