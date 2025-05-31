#!/usr/bin/env python
# coding: utf-8

# In[18]:


import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt


# In[19]:


df = pd.read_csv(r'C:\clean_NYC_v1.csv')
df


# In[20]:


df.drop(df[df['days_since_review']==9999].index, inplace = True)


# In[21]:


df.drop(columns=['Unnamed: 0'], inplace = True)


# In[56]:


df.columns


# In[35]:


from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor


# In[92]:


X = df[['neighbourhood_group_Bronx','neighbourhood_group_Brooklyn', 'neighbourhood_group_Manhattan',
        'neighbourhood_group_Queens', 'neighbourhood_group_Staten Island','room_type_Entire home/apt','room_type_Private room',
        'room_type_Shared room','minimum_nights','calculated_host_listings_count']]
y = df['log_price']


# In[99]:


X = df[['number_of_reviews', 'reviews_per_month', 'minimum_nights',
        'latitude', 'longitude', 'availability_365', 'calculated_host_listings_count'] +
       [col for col in df.columns if col.startswith('neighbourhood_group_')] +
       [col for col in df.columns if col.startswith('room_type_')]]


# ### number of reviews 넣은 것 보다 그냥 인코딩 한 것끼리 예측한게 더 정확도 높다

# In[100]:


X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)


# In[101]:


model = RandomForestRegressor(n_estimators = 100, random_state=42)
model.fit(X_train, y_train)


# In[102]:


y_pred = model.predict(X_test)

plt.scatter(y_test, y_pred, alpha=0.3)
plt.xlabel("True Log Price")
plt.ylabel("Predicted Log Price")
plt.title("Prediction vs True Values")
plt.show()



# In[103]:


model.score(X_test, y_test)


# In[104]:


cv_score = cross_val_score(model, X,y,cv=10)


# In[105]:


print(cv_score)
print(cv_score.mean())


# In[75]:


df.corr(numeric_only = True)

