#!/usr/bin/env python
# coding: utf-8

# # Predicting Future Sales
# 
# # Name: Rohith Kumar Reddy Kethireddy
# 
# # CS551A Final project
# 
# 

# ## Importing Libraries

# In[106]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
get_ipython().run_line_magic('matplotlib', 'inline')
# Read the data
sales_train = pd.read_csv('sales_train.csv')
items = pd.read_csv('items.csv')
item_categories = pd.read_csv('item_categories.csv')
shops = pd.read_csv('shops.csv')
test = pd.read_csv('test.csv')


# ## Reading  data

# In[107]:


sales_train.head()


# In[108]:


items.head()


# In[109]:


item_categories.head()


# In[110]:


shops.head()


# In[111]:


test.head()


# ## Exploratory data Analysis

# In[112]:


# check for missing values present in the data
sales_train.isnull().sum()


# In[113]:


items.isnull().sum()


# In[114]:


item_categories.isnull().sum()


# In[115]:


shops.isnull().sum()


# In[116]:


# Most sold items along with their names
most_sold = sales_train.groupby('item_id')['item_cnt_day'].sum().sort_values(ascending=False)
# join most_sold with items to get the names of the items
most_sold = most_sold.to_frame().join(items.set_index('item_id'), how='left')
most_sold.head(10)


# In[117]:


# Costliest items along with their names
costliest = sales_train.groupby('item_id')['item_price'].mean().sort_values(ascending=False)
# join costliest with items to get the names of the items
costliest = costliest.to_frame().join(items.set_index('item_id'), how='left')
costliest.head(10)


# In[118]:


# shops which sold the most
most_sold_shops = sales_train.groupby('shop_id')['item_cnt_day'].sum().sort_values(ascending=False)
# join most_sold_shops with shops to get the names of the shops
most_sold_shops = most_sold_shops.to_frame().join(shops.set_index('shop_id'), how='left')
most_sold_shops.head(10)


# In[119]:


# Number of unique items sold
len(sales_train['item_id'].unique())


# In[120]:


# Number of unique shops
len(sales_train['shop_id'].unique())


# In[121]:


# Number of unique item categories
len(items['item_category_id'].unique())


# In[122]:


# Total sales done per month
sales_per_month = sales_train.groupby('date_block_num')['item_cnt_day'].sum()
sales_per_month.plot(kind='bar', figsize=(15, 5), title='Total sales per month')


# In[123]:


# Adding item_category_id column to sales_train
sales_train = sales_train.join(items.set_index('item_id'), on='item_id', how='left')
sales_train.head()


# In[124]:


# Removing the outliers
sales_train = sales_train[(sales_train['item_price'] > 0) & (sales_train['item_price'] < 300000)]
sales_train = sales_train[(sales_train['item_cnt_day'] > 0) & (sales_train['item_cnt_day'] < 1000)]

sales_train.head()


# In[125]:


# drop item_name column
name = sales_train['item_name']
sales_train.drop('item_name', axis=1, inplace=True)

# drop date column
date = sales_train['date']
sales_train.drop('date', axis=1, inplace=True)


# # Building a Model

# In[126]:


# split sales_train into train and test using train_test_split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(sales_train.drop(['item_cnt_day'], axis=1), sales_train['item_cnt_day'], test_size=0.2, random_state=42)


# In[127]:


# create a validation set from train
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)


# In[128]:


# k nearest neighbours 
from sklearn.neighbors import KNeighborsRegressor
knn = KNeighborsRegressor(n_neighbors=5)
knn.fit(X_train, y_train)
knn.score(X_val, y_val)


# - Accuracy obtained is 11%.
# checking if there is any improvement by preprocessing the data
# 
# 
# ## Improvements to the Model

# In[129]:


# add date column to sales_train
#checking if there is qn mprovement
sales_train['date'] = date


# In[130]:


# Removing duplicate data
def drop_duplicate(data, subset):
    print('Shape before drop:', data.shape)
    before = data.shape[0]
    data.drop_duplicates(subset,keep='first', inplace=True) #subset is list where you have to put all column for duplicate check
    data.reset_index(drop=True, inplace=True)
    print('Shape after drop:', data.shape)
    after = data.shape[0]
    print('Total Duplicate:', before-after)

subset = ['date', 'date_block_num', 'shop_id', 'item_id','item_cnt_day']
drop_duplicate(sales_train, subset = subset)


# In[131]:


#Creating different categories for the data in the item_category

l = list(item_categories.item_category_name)
list_categories = l

for i in range(1,8):
    list_categories[i] = 'Access'

for i in range(10,18):
    list_categories[i] = 'Consoles'

for i in range(18,25):
    list_categories[i] = 'Consoles Games'

for i in range(26,28):
    list_categories[i] = 'phone games'

for i in range(28,32):
    list_categories[i] = 'CD games'

for i in range(32,37):
    list_categories[i] = 'Card'

for i in range(37,43):
    list_categories[i] = 'Movie'

for i in range(43,55):
    list_categories[i] = 'Books'

for i in range(55,61):
    list_categories[i] = 'Music'

for i in range(61,73):
    list_categories[i] = 'Gifts'

for i in range(73,79):
    list_categories[i] = 'Soft'

item_categories['cats'] = list_categories
item_categories.head()


# In[132]:


#Adding column naed date
sales_train['date'] = pd.to_datetime(sales_train['date'], format='%d.%m.%Y')
sales_train.head()


# In[133]:


# Pivot by month to wide format
p_df = sales_train.pivot_table(index=['shop_id','item_id'], columns='date_block_num', values='item_cnt_day',aggfunc='sum').fillna(0.0)
p_df.head()


# In[134]:


# Join with categories
sales_train_cleaned = p_df.reset_index()
sales_train_cleaned['shop_id']= sales_train_cleaned.shop_id.astype('str')
sales_train_cleaned['item_id']= sales_train_cleaned.item_id.astype('str')
item_to_cat_df = items.merge(item_categories[['item_category_id','cats']], how="inner", on="item_category_id")[['item_id','cats']]
# Converting item_id to string
item_to_cat_df['item_id'] = item_to_cat_df['item_id'].astype('str')
sales_train_cleaned = sales_train_cleaned.merge(item_to_cat_df, how="inner", on="item_id")


# In[135]:



from sklearn import preprocessing

number = preprocessing.LabelEncoder()
sales_train_cleaned['cats'] = number.fit_transform(sales_train_cleaned.cats)
sales_train_cleaned = sales_train_cleaned[['shop_id', 'item_id', 'cats'] + list(range(34))]
sales_train_cleaned.head()


# ## Modelling 

# In[136]:


import sys
get_ipython().system('{sys.executable} -m pip install xgboost')


# In[137]:


import xgboost as xgb


# In[138]:


# All the data points are plotted by the KNN model, which then examines which data points most closely resemble the one we're looking for.
# KNN model will take all of the features in the data and attempt to forecast using all of the features.
# This will result in inaccurate forecasts since there will be too many dimensions for the model to decide which dimension is the "nearest neighbor." Consequently, we will use the decision-tree based XGBoost model.
# XGBoost is a decision tree based model that uses gradient boosting.Gradient boosting is a method for merging several weak models to get a strong model.
#The decision tree-based XGBoost model constructs decision trees one after the other while learning from the flaws of the preceding model.
# XGBoost model

param = {'max_depth':10, 
         'subsample':1,
         'min_child_weight':0.5,
         'eta':0.3, 
         'num_round':1000, 
         'seed':1,
         'silent':0,
         'eval_metric':'rmse'}

progress = dict()

# create a DMatrix
xgbtrain = xgb.DMatrix(sales_train_cleaned.iloc[:,  (sales_train_cleaned.columns != 33)].values, sales_train_cleaned.iloc[:, sales_train_cleaned.columns == 33].values)
watchlist  = [(xgbtrain,'train-rmse')]

# train the model
bst = xgb.train(param, xgbtrain)
preds = bst.predict(xgb.DMatrix(sales_train_cleaned.iloc[:,  (sales_train_cleaned.columns != 33)].values))

# plot the feature importance
xgb.plot_importance(bst)


# In[140]:


df = test.copy()
df['shop_id'] = df.shop_id.astype('str')
df['item_id'] = df.item_id.astype('str')
df = df.merge(sales_train_cleaned, how="left", on=["shop_id", "item_id"]).fillna(0.0)
df.head()


# In[141]:


# Since we are making predictions for the upcoming month or the future, forward the prediction forward by one month.
d = dict(zip(df.columns[4:],list(np.array(list(df.columns[4:])) - 1)))

df  = df.rename(d, axis = 1)
preds = bst.predict(xgb.DMatrix(df.iloc[:, (df.columns != 'ID') & (df.columns != -1)].values))


# In[142]:


# Normalize prediction to [0-20]
preds = list(map(lambda x: min(20,max(x,0)), list(preds)))
sub_df = pd.DataFrame({'ID':df.ID,'item_cnt_month': preds })
sub_df.describe()


# In[143]:


sub_df.to_csv('project.csv',index=False)

