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


# In[2]:


from nispat.normative_model.norm_utils import norm_init


# ## Read in data

# In[3]:


df_pheno = pd.read_csv('data/df_pheno.csv')
df_pheno.set_index(['bblid','scanid'], inplace = True)
df_pheno.head()


# In[4]:


df_system = pd.read_csv('data/df_system.csv')
df_system.set_index(['bblid','scanid'], inplace = True)
df_system.head()


# First, we'll pull out some training data

# In[5]:


metric = 'jd'
X = df_pheno.loc[df_pheno['squeakycleanExclude'] == 0, 'ageAtScan1_Years'].values
Y = df_system.loc[df_pheno['squeakycleanExclude'] == 0, metric].values
if len(X.shape) == 1: X = X[:, np.newaxis]
if len(Y.shape) == 1: Y = Y[:, np.newaxis]


# And do a quick linear plot of training data

# In[6]:


sns.set(style='white', context = 'talk', font_scale = 0.8)
f = sns.jointplot(x = X, y = Y, kind = 'reg')
f.ax_joint.set_xlabel('Age')
f.ax_joint.set_ylabel('Brain feature')
f.annotate(sp.stats.pearsonr)
f.plot_joint(plt.scatter, c = "k", s = 5, linewidth = 2, marker = ".", alpha = 0.3)
f.ax_joint.collections[0].set_alpha(0)


# Standardize training data

# In[7]:


mX = np.mean(X, axis=0)
sX = np.std(X,  axis=0)
Xz = (X - mX) / sX

mY = np.mean(Y, axis=0)
sY = np.std(Y, axis=0)
Yz = (Y - mY) / sY


# Create evenly spaced X values for prediction

# In[8]:


# Range of X
X_range = [np.min(X), np.max(X)]
Xsy = np.arange(X_range[0],X_range[1],1)
Xsy = Xsy.reshape(-1,1)
# Standardize using training data params
Xsyz = (Xsy - mX) / sX


# Train gaussian process regression and generate predictions

# In[9]:


nm = norm_init(Xz, Yz, alg='gpr', configparam=None)
Hyp = nm.estimate(Xz, Yz)
yhat, s2 = nm.predict(Xz, Yz, Xsyz, Hyp)
Yhat = yhat * sY + mY # get the predictions back in original (unstandardized) units
nlZ = nm.neg_log_lik
S2 = s2 * sY**2 # get predictive variance


# Plot predictions and predictive variance

# In[10]:


f, axes = plt.subplots(1,1)
f.set_figwidth(5)
f.set_figheight(5)

axes.plot(Xsy, Yhat, linestyle = 'solid', color = 'b', linewidth = 1.5)
upper_bound = Yhat.reshape(-1) + np.sqrt(S2); lower_bound = Yhat.reshape(-1) - np.sqrt(S2)
axes.fill_between(Xsy.reshape(-1), lower_bound, upper_bound, alpha = 0.3, color = 'b')
axes.set_ylabel('GPR predictions')
axes.set_xlabel('Age')


# Get some test data

# In[11]:


Xte = df_pheno.loc[df_pheno['squeakycleanExclude'] == 1, 'ageAtScan1_Years'].values
Yte = df_system.loc[df_pheno['squeakycleanExclude'] == 1, metric].values
testids = range(X.shape[0], X.shape[0]+Xte.shape[0])
if len(Xte.shape) == 1: Xte = Xte[:, np.newaxis]
if len(Yte.shape) == 1: Yte = Yte[:, np.newaxis]


# Standardize using training params

# In[12]:


Xtez = (Xte - mX) / sX
Ytez = (Yte - mY) / sY


# In[13]:


yhat_test, s2_test = nm.predict(Xz, Yz, Xtez, Hyp)
Yhat_test = yhat_test * sY + mY # get the predictions back in original (unstandardized) units
S2_test = s2_test * sY**2 # get predictive variance


# Calculate deviations

# In[14]:


Z = (Yte.reshape(-1) - Yhat_test.reshape(-1)) / np.sqrt(S2_test)


# Plot Z deviations against something of interest

# In[15]:


f = sns.jointplot(x = df_pheno.loc[df_pheno['squeakycleanExclude'] == 1, 'Fear'], y = Z, kind = 'reg')
f.ax_joint.set_ylabel('Deviations (Z-scores)')
f.annotate(sp.stats.pearsonr)
f.plot_joint(plt.scatter, c = "k", s = 5, linewidth = 2, marker = ".", alpha = 0.3)
f.ax_joint.collections[0].set_alpha(0)


# Compare against original brain feature values

# In[16]:


f = sns.jointplot(x = df_pheno.loc[df_pheno['squeakycleanExclude'] == 1, 'Fear'], y = Yte.reshape(-1), kind = 'reg')
f.ax_joint.set_ylabel('Brain feature')
f.annotate(sp.stats.pearsonr)
f.plot_joint(plt.scatter, c = "k", s = 5, linewidth = 2, marker = ".", alpha = 0.3)
f.ax_joint.collections[0].set_alpha(0)

