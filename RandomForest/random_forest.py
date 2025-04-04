#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.ensemble import RandomForestClassifier
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_linnerud
data = load_linnerud()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.DataFrame(data.target, columns=data.target_names)
X_selected = y[['Weight', 'Waist', 'Pulse']]
y_target = X['Situps']
X_train, X_test, y_train, y_test = train_test_split(X_selected, y_target, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=10, max_depth=5, random_state=42)
model.fit(X_train, y_train)
joblib.dump(model, "random_forest_model.pkl")
print("Random Forest model saved successfully!")


# In[ ]:




