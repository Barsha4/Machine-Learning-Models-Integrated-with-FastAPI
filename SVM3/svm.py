#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pickle
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler


# In[5]:


# Load Wine dataset
wine = datasets.load_wine()
X = wine.data
y = wine.target


# In[7]:


# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[9]:


# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# In[11]:


# Train SVM model
model = SVC(kernel='linear', probability=True)
model.fit(X_train, y_train)


# In[13]:


# Save model & scaler using pickle
with open("svm_model.pkl", "wb") as model_file:
    pickle.dump((scaler, model), model_file)

print("Model saved as svm_model.pkl")


# In[ ]:




