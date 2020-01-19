#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os, sys
import pandas as pd
import numpy as np
import numpy.matlib
import scipy as sp
from pygam import LinearGAM, s, l

# Plotting
import matplotlib.pyplot as plt
import seaborn as sns


# ## Read in data

# In[2]:


df_pheno = pd.read_csv('data/df_pheno.csv')
df_pheno.set_index(['bblid','scanid'], inplace = True)
df_pheno.head()


# In[3]:


df_system = pd.read_csv('data/df_system.csv')
df_system.set_index(['bblid','scanid'], inplace = True)
df_system.head()


# Get some data

# In[4]:


metric = 'jd'
X = df_pheno.loc[:, 'ageAtScan1_Years']
Y = df_system.loc[:, metric]


# Estimate GAM with spline

# In[5]:


gam = LinearGAM(s(0)).fit(X, Y)
gam.gridsearch(X, Y)


# Plot

# In[6]:


XX = gam.generate_X_grid(term=0)
pdep, confi = gam.partial_dependence(term=0, X=XX, width=0.95)

plt.figure()
plt.plot(XX, pdep) # fit
plt.plot(XX, confi, c='r', ls='--') # confidence interval
plt.plot(XX, gam.prediction_intervals(XX, width=.95), color='b', ls='--') # 95% prediction interval
plt.scatter(X, Y, facecolor='gray', edgecolors='none', alpha = 0.5) # data
plt.xlabel('Age')
plt.ylabel('Brain feature')
plt.show()


# In[7]:


metric = 'jd'
X = df_pheno.loc[:, ['ageAtScan1_Years','mprage_antsCT_vol_TBV']]
Y = df_system.loc[:, metric]


# Estimate GAM with spline

# In[8]:


gam = LinearGAM(s(0) + s(1)).fit(X, Y)
gam.gridsearch(X, Y)


# Plot

# In[9]:


for i, term in enumerate(gam.terms):
    if term.isintercept:
        continue

    XX = gam.generate_X_grid(term=i)
    pdep, confi = gam.partial_dependence(term=i, X=XX, width=0.95)

    plt.figure()
    plt.plot(XX[:, term.feature], pdep) # fit
    plt.plot(XX[:, term.feature], confi, c='r', ls='--') # confidence interval
    plt.scatter(X.iloc[:,i], Y, facecolor='gray', edgecolors='none', alpha = 0.5) # data
    plt.show()

