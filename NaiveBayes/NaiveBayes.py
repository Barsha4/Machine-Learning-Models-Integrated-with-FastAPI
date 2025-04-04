#!/usr/bin/env python
# coding: utf-8

# In[7]:


import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.datasets import fetch_openml


# In[17]:


# Load dataset (Adult Income dataset from OpenML)
data = fetch_openml(name='adult', version=2, as_frame=True)
df = data.frame.copy()  # Explicitly create a copy

# Selecting relevant features and target
df = df[['age', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week', 'class']].copy()
df['class'] = df['class'].astype(str).apply(lambda x: 1 if x == '>50K' else 0)  # Convert to string first

# Split data
X = df.drop(columns=['class'])
y = df['class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[19]:


# Train Naive Bayes model
model = GaussianNB()
model.fit(X_train, y_train)


# In[21]:


# Evaluate model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")



# In[23]:


# Save model as pickle file
with open("naive_bayes_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model saved as naive_bayes_model.pkl")


# In[ ]:




