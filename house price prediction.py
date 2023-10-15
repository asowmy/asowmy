#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#HOUSE PRICE PREDICTION USINGG MACHINE LEARNING ALGORITHM


# In[1]:


# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


# In[2]:


# Load the dataset
data = pd.read_csv("C:/Users/91638/Downloads/archive (5)/kc_house_data.csv")  # Replace 'housing_data.csv' with your dataset's file path


# In[5]:


# Assuming your dataset has column names like 'square_footage' and 'num_bedrooms'
X = data[['sqft_lot', 'bedrooms']]  # Features
y = data['price']  # Target


# In[3]:


print(data.head())


# In[6]:


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


# In[7]:


# Create a linear regression model
model = LinearRegression()


# In[8]:


# Train the model on the training data
model.fit(X_train, y_train)


# In[9]:


# Make predictions on new house data
new_house_features = [[2000, 3]]  # Example: 2000 sq. ft and 3 bedrooms
predicted_price = model.predict(new_house_features)


# In[10]:


print("Predicted Price:", predicted_price)


# In[ ]:




