#!/usr/bin/env python
# coding: utf-8

# # Multiple Linear Regression - Exercise

# You are given a real estate dataset. 
# 
# Real estate is one of those examples that every regression course goes through as it is extremely easy to understand and there is a (almost always) certain causal relationship to be found.
# 
# The data is located in the file: 'real_estate_price_size_year.csv'. 
# 
# You are expected to create a multiple linear regression (similar to the one in the lecture), using the new data. 
# 
# In this exercise, the dependent variable is 'price', while the independent variables are 'size' and 'year'.
# 
# Good luck!

# ## Import the relevant libraries

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn
seaborn.set()


# ## Load the data

# In[2]:


data = pd.read_csv("real estate price size year.csv")


# In[3]:


data


# In[4]:


data.describe


# ## Create the regression

# ### Declare the dependent and the independent variables

# In[8]:


y = data['price']
x1 = data[['size','year']]


# ### Regression

# In[9]:


x = sm.add_constant(x1)
results = sm.OLS(y,x).fit()


# In[10]:


#find summary
results.summary()


# In[ ]:




