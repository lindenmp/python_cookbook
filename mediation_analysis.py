#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os, sys
import pandas as pd
import numpy as np
import numpy.matlib
import scipy as sp

# Plotting
import matplotlib.pyplot as plt
import seaborn as sns

import statsmodels.api as sm
from pingouin import mediation_analysis


# ## Read in data

# In[2]:


df_pheno = pd.read_csv('data/df_pheno.csv')
df_pheno.set_index(['bblid','scanid'], inplace = True)
df_pheno.head()


# In[3]:


df_system = pd.read_csv('data/df_system.csv')
df_system.set_index(['bblid','scanid'], inplace = True)
df_system.head()


# ### X = scanageMonths | M = brain_t1 (yeo systems) | Y = pheno_t2

# In[4]:


pheno = 'Overall_Psychopathology'; print(pheno)
metric = 'jd'; print(metric)


# In[5]:


brain_preds = df_system.filter(regex = metric+'_')
mediators = list(brain_preds.columns)
df_input = pd.concat((df_pheno.loc[:,'ageAtScan1'],
                      brain_preds,
                      df_pheno.loc[:,pheno].rename('Y')), axis = 1) # combine
df_input.dropna(axis = 0, inplace = True)

df_input = (df_input - df_input.mean())/df_input.std() # standardize
df_input = pd.concat((df_input, df_pheno.loc[:,'sex']-1), axis = 1) # combine
df_input = sm.add_constant(df_input) # add constant term
df_input.dropna(axis = 0, inplace = True)


# In[6]:


df_input.head()


# In[7]:


med = mediation_analysis(data=df_input, x='ageAtScan1', m=mediators, y='Y', alpha=0.05, n_boot = 1000, seed=0)
med.set_index('path', inplace = True)


# In[8]:


med.loc[med['sig'] == 'Yes',:]


# In[9]:


np.round(med.filter(regex = 'Indirect', axis = 0).loc[med['sig'] == 'Yes',:], decimals=3)


# In[10]:


med.loc['Direct',:]


# In[11]:


col = metric+'_17'
np.round(med.filter(regex = col, axis = 0), decimals=3)


# In[12]:


f = sns.jointplot(x = df_input.loc[:,'ageAtScan1'], y = df_input.loc[:,col], kind="reg")
f.annotate(sp.stats.pearsonr)
f.plot_joint(plt.scatter, c = "k", s = 5, linewidth = 2, marker = ".", alpha = 0.3)
f.ax_joint.collections[0].set_alpha(0)


# In[13]:


f = sns.jointplot(x = df_input.loc[:,'ageAtScan1'], y = df_input.loc[:,'Y'], kind="reg")
f.annotate(sp.stats.pearsonr)
f.plot_joint(plt.scatter, c = "k", s = 5, linewidth = 2, marker = ".", alpha = 0.3)
f.ax_joint.collections[0].set_alpha(0)


# In[14]:


f = sns.jointplot(x = df_input.loc[:,col], y = df_input.loc[:,'Y'], kind="reg")
f.annotate(sp.stats.pearsonr)
f.plot_joint(plt.scatter, c = "k", s = 5, linewidth = 2, marker = ".", alpha = 0.3)
f.ax_joint.collections[0].set_alpha(0)

