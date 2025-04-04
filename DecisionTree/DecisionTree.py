#!/usr/bin/env python
# coding: utf-8

# In[37]:


import pickle
import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor


# In[39]:


# Load dataset (Diabetes dataset from sklearn)
data = load_diabetes()
df = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target


# In[41]:


# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.2, random_state=42)


# In[47]:


# Train Decision Tree model
model = DecisionTreeRegressor()
model.fit(X_train, y_train)


# In[49]:


# Save model to a pickle file
with open("decision_tree_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model training complete. Pickle file saved.")


# In[ ]:




