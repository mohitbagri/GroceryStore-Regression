import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor

# Reading data from the CSV file
train_data=pd.read_csv("C:/Users/MOHIT/Desktop/AV/Train-Reg.csv")
test_data=pd.read_csv("C:/Users/MOHIT/Desktop/AV/Test-Reg.csv")
data = pd.concat([train_data,test_data], ignore_index=True)
df2=data.copy()

# Data parsing and processing
df2['Item_MRP']=np.log(df2['Item_MRP'])

df2['Item_Weight']=df2['Item_Weight'].fillna(df2['Item_Weight'].mean())

df2.loc[df2['Item_Visibility']==0.0,'Item_Visibility']=df2['Item_Visibility'].mean()
df2['Item_Visibility']=df2['Item_Visibility']*100

df2['Outlet_Establishment_Year']=2013-df2['Outlet_Establishment_Year']

df2['Item_Fat_Content'].replace("LF",0,inplace=True)
df2['Item_Fat_Content'].replace("Low Fat",0,inplace=True)
df2['Item_Fat_Content'].replace("low fat",0,inplace=True)
df2['Item_Fat_Content'].replace("Regular",1,inplace=True)
df2['Item_Fat_Content'].replace("reg",1,inplace=True)

non_food=['Household','Others',0]
df2.loc[df2['Item_Type'].isin(non_food),'Item_Type']=0
df2.loc[~df2['Item_Type'].isin(non_food),'Item_Type']=1
df2.loc[df2['Item_Type']==0,'Item_Fat_Content']=2

df2['Outlet_Size']=df2['Outlet_Size'].fillna(df2['Outlet_Size'].mode()[0])
df2["Outlet_Size"] = df2["Outlet_Size"].astype('category')
df2["Outlet_Size"] = df2["Outlet_Size"].cat.codes

df2["Outlet_Location_Type"] = df2["Outlet_Location_Type"].astype('category')
df2["Outlet_Location_Type"] = df2["Outlet_Location_Type"].cat.codes
df2["Outlet_Type"] = df2["Outlet_Type"].astype('category')
df2["Outlet_Type"] = df2["Outlet_Type"].cat.codes

df2['Item_Outlet_Sales']=np.log(df2['Item_Outlet_Sales'])

df2.loc[df2['Item_Visibility']>25,'Item_Visibility']=df2['Item_Visibility'].mean()

# Train test split
drop_columns=['Item_Outlet_Sales','ID']
y=df2['Item_Outlet_Sales'].dropna()
x=df2[:8523].drop(drop_columns,axis=1)
x_validation=df2[8523:].drop(drop_columns,axis=1)
x_train, x_test,y_train,y_test = train_test_split(x,y, test_size=0.25, random_state=42)

# Random Forest regressor
reg=RandomForestRegressor(n_estimators= 400, min_samples_split= 2, min_samples_leaf= 4, max_features='sqrt', max_depth= 10, bootstrap=True,random_state=42)

# Hyper parameter tuning
'''
# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]
# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}

rf_random = RandomizedSearchCV(estimator = reg, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = 1)
rf_random.fit(x_train, y_train)
print(rf_random.best_params_)
'''

# Fitting the model and predicting test data values
reg.fit(x_train,y_train)
y_predict=reg.predict(x_train)
print(reg.score(x_test,y_test))
print(mean_squared_error(y_train, y_predict))
