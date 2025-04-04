#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_diabetes


# In[3]:


# Load dataset
data = load_diabetes()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target


# In[5]:


# Define features and target
X = df.drop(columns=['target'])
y = df['target']


# In[7]:


# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[9]:


# Train model
model = LinearRegression()
model.fit(X_train, y_train)


# In[11]:


# Save model
with open("linear_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model trained and saved as 'linear_model.pkl'")


# In[ ]:




