# SGD-Regressor-for-Multivariate-Linear-Regression

## AIM:
To write a program to predict the price of the house and number of occupants in the house with SGD regressor.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
```
STEP 1.Start
STEP 2. Data preparation
STEP 3. Hypothesis Definition
STEP 4. Cost Function
STEP 5. Parameter Update Rule
STEP 6. Iterative Training
STEP 7. Model evaluation
STEP 8. End
```
## Program:
```
/*
Program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor.
Developed by: G SANJAY
RegisterNumber:  212224230243
*/
```
```
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import SGDRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
#Load the California Housing dataset
dataset=fetch_california_housing()
df=pd.DataFrame(dataset.data,columns=dataset.feature_names)
df['HousingPrice']=dataset.target
print(df.head())
#Use the first 3 features as inputs
x=df.drop(columns=['AveOccup','HousingPrice'])#data[:,:3]#Features: 'MedInc','HouseAge','AveRooms'
#Use 'MedHouseVal' and 'AveOccup' as output variables
y=df[['AveOccup','HousingPrice']]#np.column_stack((data.target,data.data[:,6]))#Targets: 'MedHouseVal','AveOccup'
#Split the data into training and testing sets
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
#scale the features and target variables
scaler_x=StandardScaler()
scaler_y=StandardScaler()
x_train=scaler_x.fit_transform(x_train)
x_test=scaler_x.transform(x_test)
y_train=scaler_y.fit_transform(y_train)
y_test=scaler_y.transform(y_test)
#Initialize the SGDRegressor
sgd=SGDRegressor(max_iter=1000,tol=1e-3)
#Use MultiOutputRegressor to handle multiple output variables
multi_output_sgd=MultiOutputRegressor(sgd)
#Train the model
multi_output_sgd.fit(x_train,y_train)
#predict on the test data
y_pred=multi_output_sgd.predict(x_test)
#Inverse transform the predictions to get them back to the original scale
y_pred=scaler_y.inverse_transform(y_pred)
y_test=scaler_y.inverse_transform(y_test)
#Evaluate the model using Mean Squared Error
mse=mean_squared_error(y_test,y_pred)
print("Mean Squared Error:",mse)
#Optionally,print some predictions
print("\nPredictions:\n",y_pred[:5])#Print first 5 predictions
```

## Output:
![image](https://github.com/user-attachments/assets/4192e1ce-36f2-426c-8b27-0c74a9beea81)


## Result:
Thus the program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor is written and verified using python programming.
