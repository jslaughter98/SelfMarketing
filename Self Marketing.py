#!/usr/bin/env python
# coding: utf-8

# In[6]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.api as sm


# In[4]:


# Set up random data
np.random.seed(123)
n = 100
GDP = np.random.uniform(10000, 1000000, n)
true_intercept = 5000
true_slope = 0.1
noise = np.random.normal(0, 3000, n)
USEUR = true_intercept = + true_slope * GDP + noise

data = pd.DataFrame({'GDP': GDP, 'USEUR': USEUR})
print(data.head())


# In[8]:


# Scatter Plot
plt.figure(figsize=(10, 6))
sns.scatterplot(data=data, x='GDP', y='USEUR')
plt.title("Scatter Plot of GDP vs. USEUR")
plt.xlabel("GDP")
plt.ylabel("USEUR")
plt.show()


# In[9]:


# Fit linear regression
x = sm.add_constant(data['GDP'])
y = data['USEUR']
model = sm.OLS(y, x).fit()
print(model.summary())
