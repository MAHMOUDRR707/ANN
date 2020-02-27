#!/usr/bin/env python
# coding: utf-8

# In[19]:


#importing libraures
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt


# In[20]:


#data procerssing
data=pd.read_csv("Churn_Modelling.csv")
x=data.iloc[:,3:13].values
y=data.iloc[:,13].values


# In[22]:


#categorical data
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelencoder=LabelEncoder()
x[:,1]=labelencoder.fit_transform(x[:,1])
x[:,2]=labelencoder.fit_transform(x[:,2])
onehotencoder=OneHotEncoder(categorical_features=[1])
x=onehotencoder.fit_transform(x).toarray()
#to ignore dummy trap
x=x[:,1:]


# In[23]:


#training , splitting and testing the data
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.25,random_state=0)


# In[24]:


#feature scalling
from sklearn.preprocessing import StandardScaler
standardscaler=StandardScaler()
x_train=standardscaler.fit_transform(x_train)
x_test=standardscaler.transform (x_test)


# In[28]:


#build ANN
#import libraries
import keras
from keras.models import Sequential
from keras.layers import Dense 


# In[31]:


# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu', input_dim = 11))

# Adding the second hidden layer
classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu'))

# Adding the output layer
classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
classifier.fit(x_train, y_train, batch_size = 10, nb_epoch = 100)


# In[34]:


#predict the result
y_pred=classifier.predict(x_test)
y_pred=(y_pred>.5)

#scalling
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)


# In[35]:


print(cm)


# In[ ]:




