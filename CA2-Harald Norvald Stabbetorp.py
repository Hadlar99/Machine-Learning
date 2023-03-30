#!/usr/bin/env python
# coding: utf-8

# # CA02 - Training Perceptron and Adaline models

# Make sure you: a) describe briefly what you intend to do using markdown cells; b) comment your code properly but briefly, such that the reader can easily understand what the code is doing.
# 

# ## Imports

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Import the relevant classes from adaline.py and perceptron.py 
#in the classifiers folder
from classifiers.perceptron import Perceptron
from classifiers.adaline import AdalineGD


# ## Loading and exploring data
# 
# 
# Visualise the raw data with appropriate plots and inspect it for possible outliers or inconsistencies. Comment briefly on what you see and how this will impact the performance of the perceptron and adaline. For this use no more than three sentences.

# In[2]:


# Insert your code below
# ======================
#Loads the file
raw_df = pd.read_csv('assets\Wine.csv')
#Visualise the date in differetn ways
display(raw_df.head())
sns.pairplot(raw_df, hue='class')
plt.figure()
raw_df.plot(kind='box', 
                 subplots=True, figsize=(15,10) )
plt.show()


# By looking at the boxplot we can se thar residual sugar, chlorides, sulfur dioxide and sulphates has a lot of outliers but in total we are working with a complete data set without any missing values. The data also looks like it has a lot of collinearity which most likely will make the performance worse.

# ## Preprocessing data and training models
# 
# - Split the data into training and test_sets, where 400 samples are used for training
# - Make sure the target values are appropriate for the Adaline and Perceptron classifiers
# 
# With each of the 400 models, you should predict the classes of the unseen samples in the test data and compute the test set classification accuracy. Store the results in a (8 x 50) numpy array or a pandas dataframe.

# __Preprocessing:__

# In[3]:


# Insert your code below
# ======================
#converts the class from 0 to -1
raw_df['class'] = raw_df['class'].replace(to_replace=0,value=-1)
#splits the data into training and test
X_train = raw_df.iloc[0:400, :-1].values
y_train = raw_df.iloc[0:400, -1].values
X_test = raw_df.iloc[400:, :-1].values
y_test = raw_df.iloc[400:, -1].values

# Scales the data by using the mean and std for X_train
X_train_sc = (X_train - np.mean(X_train, axis=0)) / np.std(X_train, axis=0, ddof=1)
X_test_sc = (X_test - np.mean(X_train, axis=0)) / np.std(X_train, axis=0, ddof=1)    


# Checks if X_train is scaled correctly
X_train_sc_mean = np.mean(X_train_sc, axis=0)
X_train_sc_std = np.std(X_train_sc, axis=0, ddof=1)
display(X_train_sc_mean)
display(X_train_sc_std)

# Checks if X_test is scaled correctly
X_test_sc_mean = np.mean(X_test_sc, axis=0)
X_test_sc_std = np.std(X_test_sc, axis=0, ddof=1)
display(X_test_sc_mean)
display(X_test_sc_std)


# __Training:__

# In[4]:


# Insert your code below
# ======================
#List to store performance for subset
TotalPerceptron = []
#List to store rownames
rowname_Perc = []
for subset in range(50, 401, 50): 
    #Creates rownames
    rowname_Perc += ["{0} out of 400".format(subset)]
    #List to store accuracy
    AccuracyPerc = []
    #List to store column name
    columnname_Perc = []
    for epoch in range(1,51):
        #activates the function
        perp = Perceptron(n_iter=epoch, eta=0.0001)
        #trains the model
        perp.fit(X_train_sc[:subset], y_train[:subset])
        #predicts the outcome
        y_pred_perp = perp.predict(X_test_sc)
        
        #create columnames
        columnname_Perc += ["{0} epoch".format(epoch)]
        
        #calculate accuracy and puts it in list
        correct = 0
        for j in range(len(y_pred_perp)):    
            if y_test[j] == y_pred_perp[j]:
                correct += 1
        acc = correct / len(y_pred_perp) * 100
        AccuracyPerc += [acc]
        
    TotalPerceptron += [AccuracyPerc]
    
#makes it a datframe and changes index and column names    
df_Perc = pd.DataFrame(TotalPerceptron)
df_Perc.columns = columnname_Perc 
df_Perc.index = rowname_Perc
df_Perc.head(8)


# In[5]:


#List to store performance for subset
TotalAdaline = []
#List for rownames
rowname_Ada = []
for subset in range(50, 401, 50):
    #Create rownames
    rowname_Ada += ["{0} out of 400".format(subset)]
    #List to store accuracy
    AccuracyAda = []
    #List for columnames
    columnname_Ada = []
    for epoch in range(1,51):
        #Activates the model
        ada = AdalineGD(n_iter=epoch, eta=0.0001)
        #Train the model
        ada.fit(X_train_sc[:subset], y_train[:subset])
        #Predict outcome
        y_pred_ada = ada.predict(X_test_sc)
        
        #Make columnames
        columnname_Ada += ["{0} epoch".format(epoch)]
        
        #Calculate the performance
        correct = 0
        for j in range(len(y_pred_ada)):    
            if y_test[j] == y_pred_ada[j]:
                correct += 1
        acc = correct / len(y_pred_ada) * 100
        AccuracyAda.append(acc)
        
    TotalAdaline.append(AccuracyAda)

#Makes a datafram and changes index and column names
df_Ada = pd.DataFrame(TotalAdaline)
df_Ada.columns = columnname_Ada
df_Ada.index = rowname_Ada
df_Ada.head(8)


# ## Visualizing results
# 
# Plot a heatmap of the results (test set classification accuracy) using Python plotting packages matplotlib or seaborn (Lenker til en ekstern side.). See below what the heatmaps should look like for the two classification algorithms.

# In[6]:


# Insert your code below
# ======================
#Create figure size and plot heatmap
figure = plt.gcf()
figure.set_size_inches(10,7)
sns.heatmap(df_Perc)
plt.title("Perceptron")
plt.figure()

#Create figure size and plot heatmap
figure = plt.gcf()
figure.set_size_inches(10,7)
sns.heatmap(df_Ada)
plt.title("Adaline")
plt.show()


# ## Metrics

# Provide the maximum test set classification accuracy for each, the perceptron classifier and the adaline classifier and information on with which combination of number training data samples and number of epochs the best classification accuracy was achieved. 

# In[7]:


# Insert your code below
# ======================
#Find the max value
max_val_Perc = df_Perc.max().max()
#Find position
row_index_Perc, col_index_Perc = np.where(df_Perc == max_val_Perc)
print("The maximum value for the Perceptron:\n",
      df_Perc.iloc[row_index_Perc, col_index_Perc],"\n")

#Find max value
max_val_Ada = df_Ada.max().max()
#Find postion
row_index_Ada, col_index_Ada = np.where(df_Ada == max_val_Ada)
print("The maximum value for the Adaline: \n",
      df_Ada.iloc[row_index_Ada, col_index_Ada])


# ## Discussion

# The training time of the simpler perceptron algorithm is quite a bit longer than the training time of the adaline algorithm. What might be the reason for this?

# __Insert your answer below__
# 
# =====================
# 
# There is a multiple of reasons to why the perceptron is slower tha the adaline algorithm. The perceptron learning rule that only updates the weight when an error is made makes it slower beacuse of the data is highly colinear and hard to sepereate. The perceptron will then use multiple tries on one sample with the step function if it gets misclassified wich will increase the calculation time for each epoch.
# 
# The adaline algorithm will instead update the weight after each epoch with the gradient decent wich will lead to faster calculation time.
