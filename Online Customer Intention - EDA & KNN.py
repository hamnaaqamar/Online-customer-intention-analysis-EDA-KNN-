#!/usr/bin/env python
# coding: utf-8

# In[44]:


import opendatasets as od
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import seaborn as sns
sns.set_theme()
sns.set_style('whitegrid')
sns.set_palette(['#3F7C85', '#FF5F5D', '#00CCBF', '#72F2EB', '#747E7E'])


# In[2]:


od.download("https://www.kaggle.com/datasets/henrysue/online-shoppers-intention/data")


# In[3]:


import os


# In[4]:


data_dir = '.\online-shoppers-intention'


# In[5]:


os.listdir(data_dir)


# In[45]:


df = pd.read_csv('online_shoppers_intention.csv')
df


# ### Exploratory Data Analysis

# In[17]:


#display all columns
pd.pandas.set_option('display.max_columns',None)
df.head()


# In[10]:


#shape of dataset
df.shape


# In[16]:


df.info()


# #### checking for missing values

# In[14]:


df.isnull().sum()


# In[26]:


df.describe()


# #### Data Cleaning and minipulation

# In[27]:


df.groupby('Month').agg('count')


# In[28]:


# changing June to Jun 
df.loc[df['Month'] == 'June', 'Month'] = 'Jun'


# In[29]:


month_order = ['Feb','Mar', 'May', 'Jun', 
               'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
df['Month'] = pd.Categorical(df['Month'] , categories = month_order , ordered = True)


# In[30]:


df.groupby('Month').agg('count')


# In[46]:


# list of numerical variables
numerical_features = [feature for feature in df.columns if df[feature].dtype != 'O' and df[feature].dtype != bool]
print('Number of numerical variables: ', len(numerical_features))

df[numerical_features].head()


# In[20]:


#continous and discrete variables in the data
discrete_feature = [feature for feature in numerical_features if len(df[feature].unique())<25]
print("Discrete Variables Count: {}".format(len(discrete_feature)))


# In[21]:


discrete_feature


# In[23]:


df[discrete_feature].head()


# In[56]:


categorical_features = []
for x in features:
    if len(df[x].value_counts()) < 20:
        categorical_features.append(x)
categorical_features


# In[31]:


# creating the density plot for numerical columns
df[numerical_features].plot(kind='density',subplots=True,layout=(4,4),sharex=False,figsize=(25,12))


# In[35]:


# Set up the layout for subplots
fig, axes = plt.subplots(nrows=5, ncols=4, figsize=(18, 15))
fig.suptitle('Distribution of Numerical Features', y=1.02)

# Plotting histograms for each numerical feature
for i, feature in enumerate(numerical_features):
    if i < len(axes.flat):
        sns.histplot(df, x=feature, bins=20, kde=True, ax=axes.flat[i], hue='Revenue', palette='viridis')
        axes.flat[i].set_title(feature)
        axes.flat[i].set_xlabel('')
        axes.flat[i].set_ylabel('')
# Show the plot
plt.show()


# Observations:
# 1) Region graph reveals that certain regions have higher revenue generation as compared to others
# 2) operating systems graph shows that usage of various operating systems by website visitors is a factor for determining whether revenue is generated on each operating system or not. 
# 3)browsers graph shows that certain browsers may have an impact on revenue

# #### Revenue Analysis to determine if Revenue generation is influenced by specific time periods such as months, weekends, special days

# In[34]:


df.groupby('Month')['Revenue'].value_counts().unstack('Revenue').plot(kind='bar',stacked=True,figsize=(10,5))


# In[51]:


month_customer = df[df['Revenue']==True].groupby('Month')['Revenue'].agg(Buyers='count').reset_index()
month_customer['Non-Buyers']=df[df['Revenue']==False].groupby('Month')['Revenue'].agg('count').values


plt.figure(figsize=(10, 8))
ax = plt.gca()  # Get current axis 

# Plot for Buyers
buyers_color = sns.color_palette('pastel')[2]
sns.lineplot(data=month_customer, x='Month', y='Buyers', label='Buyers', color=buyers_color, marker='o', ax=ax)

# Plot for Non-Buyers
non_buyers_color = sns.color_palette('pastel')[3]
sns.lineplot(data=month_customer, x='Month', y='Non-Buyers', label='Non-Buyers', color=non_buyers_color, marker='o', ax=ax)

plt.ylabel('Session count')

plt.show()


# Observations:
# 1)from Jun-Oct Buyers and Non-buyers stay relatively constant
# 2)peak season for product purchase is in May and november which is indicated by highest session count   during this time.
# 3)lowest revenue is generated in June

# In[37]:


df.groupby('Weekend')['Revenue'].value_counts().unstack('Revenue').plot(kind ='bar', stacked = True
                                                                        , figsize = (6,6))


# In[56]:


# Group by 'Weekend' and 'Revenue', calculate counts, and pivot the table
weekend_df = df.groupby(['Weekend', 'Revenue']).size().unstack(fill_value=0)

# Rename columns and index
weekend_df.columns = ['Buyers', 'Non-Buyers']
weekend_df.index = ['Week_end', 'Week_day']

# Calculate the percentage of Buyers
weekend_df = weekend_df.assign(BuyersPct=lambda x: (x['Buyers'] / (x['Buyers'] + x['Non-Buyers'])) * 100)


# In[63]:


colors = sns.color_palette('pastel')[2:4][::-1]
plt.figure(figsize=(15, 7.5))
plt.subplot(1, 2, 2)
ax2=sns.barplot(x=weekend_df.index, 
            y=weekend_df['BuyersPct'],
            palette='pastel')
plt.ylabel("Buying percentage")
ax2.set_title("Plot of buying percentage on weekday and weekend")

plt.show()


# Observations:
# 1) Although total number of sessions are lower on weekends but probability of sessions resulting in purchases is more on weekends
# 2) can be concluded that most product searching occurs on weekdays but completion of purchase is higher on weekends

# In[38]:


df.groupby('SpecialDay')['Revenue'].value_counts().unstack('Revenue').plot(kind = 'bar', stacked = True
                                                                          ,figsize = (7,7))


# ### Analysis on the basis of Page Interaction
# types of pages: Administrative, Informational, Product Related.
# other features considered: Revenue, Visitor Type

# In[4]:


# As 'Revenue' column contains boolean values
page_count_df = df.groupby('Revenue')[['Administrative', 'Informational', 'ProductRelated']].mean().reset_index()

# Converting boolean values to strings
page_count_df['Revenue'] = np.where(page_count_df['Revenue'], 'Buyers', 'Non-Buyers')


# In[11]:


# melting the page_count_df to reshape it
melted_df = pd.melt(page_count_df, id_vars = 'Revenue', value_vars = ['ProductRelated', 'Administrative', 'Informational'],
                   var_name = 'PageType', value_name = 'Value')

#creating bar plot
plt.figure(figsize=(10, 6))
sns.barplot(data=melted_df, x='Revenue', y='Value', hue='PageType', palette='pastel')
plt.xlabel('')
plt.ylabel('Average Page view (n)')
plt.show()


# In[9]:


page_duration_df = df.groupby('Revenue')[['Administrative_Duration', 'Informational_Duration',
                                          'ProductRelated_Duration']].mean().reset_index()
page_duration_df['Revenue'] = np.where(page_duration_df['Revenue'], 'Buyers', 'Non-Buyers')


# In[12]:


# melting the page_count_df to reshape it
melted_df = pd.melt(page_duration_df, id_vars = 'Revenue', value_vars = ['ProductRelated_Duration', 'Administrative_Duration', 'Informational_Duration'],
                   var_name = 'PageType', value_name = 'Value')

#creating bar plot
plt.figure(figsize=(10, 6))
sns.barplot(data=melted_df, x='Revenue', y='Value', hue='PageType', palette='pastel')
plt.xlabel('')
plt.ylabel('Average Page visit duration (min)')
plt.show()


# Observations: 
# 1) Buyers spend greater time on product_related pages in comparison to non buyers thus engagement with product related pages is significant factor leading to purchases
# 2) buyers view informational pages whereas non buyers donot
# 3) buyers generally spend more time on all pages in comparison to non buyers

# In[19]:


VisitorType = df.groupby(['VisitorType','Revenue'])['Revenue'].agg(['count']).reset_index()
VisitorType


# In[20]:


#Analysing visitor type and revenue relation
plt.figure(figsize=(10, 5))
sns.set_style("whitegrid")
sns.barplot(x='VisitorType', y='count',hue='Revenue', data=VisitorType)
plt.title("VisitorType")
plt.xlabel("VisitorType")
plt.ylabel("Revenue")
plt.show()


# It can be observed that although Returning Visitors are more frequent but have a lower probability of making a purchase compared to New Visitors. 
# So it can be concluded, although Returning Visitors contribute to higher site traffic due to their frequent visits, they may require targeted efforts to enhance their conversion into customers. 

# #### Engagement Metrics Analysis

# In[23]:


#As Revenue is boolean- converting boolean values to strings
df_bounce_exit = df.copy()
df_bounce_exit['Revenue'] = np.where(df_bounce_exit['Revenue'], 'Buyers', 'Non-Buyers')

#setting figuire size
plt.figure(figsize=(10, 6))

#creating a scatter plot with colours based on Revenue
sns.scatterplot(data=df_bounce_exit, x='ExitRates', y='BounceRates', hue='Revenue', palette='pastel')

#adding a regression line
sns.regplot(data=df, x='ExitRates', y='BounceRates', scatter=False, color='black')

# Set axis labels, legend title, and display the plot
plt.xlabel('Exit rate')
plt.ylabel('Bounce rate')
plt.legend(title='Buyer type')
plt.show()


# Observations:
# 1) there is a positive correlation between the exit rate and the bounce rate 
# 2) the density of buyers is greater in lower exit rates which shows that lower exit rate and bounce rate results in greater probability of purchases

# In[25]:


# Condition-based counts
pg_value_revenue_true = df[(df['PageValues'] > 0) & (df['Revenue'] == True)]['Revenue'].count()
pg_nonvalue_revenue_true = df[(df['PageValues'] == 0) & (df['Revenue'] == True)]['Revenue'].count()
pg_value_revenue_false = df[(df['PageValues'] > 0) & (df['Revenue'] == False)]['Revenue'].count()
pg_nonvalue_revenue_false = df[(df['PageValues'] == 0) & (df['Revenue'] == False)]['Revenue'].count()

# Constructing the dictionary 
pg_revenue_data = {
    'Session count of page value = 0': [pg_nonvalue_revenue_true, pg_nonvalue_revenue_false],
    'Session count of page value > 0': [pg_value_revenue_true, pg_value_revenue_false]
}

# Creating the DataFrame
pg_revenue_df = pd.DataFrame(pg_revenue_data, index=['Buyers', 'Non-Buyers'])
pg_revenue_df


# In[31]:


# Calculate percentages and transpose
#x represents each column of pg_revenue_df
#x.sum() represents sum of each column
# T : This transposes the resulting DataFrame. It swaps the rows and columns, making the columns become rows and vice versa. 
#This operation rearranges the DataFrame so that the 'Page Value Category' becomes the index.
pg_revenue_df_percent = pg_revenue_df.apply(lambda x: x / x.sum() * 100).T 

# Plotting the stacked bar chart
colors = sns.color_palette('pastel')[2:4]
pg_revenue_df_percent.plot(kind='bar', stacked=True, color=colors)

# Customize legend, labels, and layout
plt.legend(title='Purchase Status', loc='upper right')
plt.xticks(rotation=0)
plt.xlabel('Page Value Category')
plt.ylabel('Buyers percentage (%)')
plt.tight_layout()

# Show the plot
plt.show()


# Observations:
# 1) If session count of page value of a user is greater than 0 then they are more likely to purchase 
# 2) session with lower page value corresponds to a lower percentage of buyers

# #### KNN 

# In[64]:


from sklearn.preprocessing import StandardScaler 
  
scaler = StandardScaler() 
  
scaler.fit(df.drop('Revenue', axis = 1)) 
scaled_features = scaler.transform(df.drop('Revenue', axis = 1)) 
  
df_feat = pd.DataFrame(scaled_features, columns = df.columns[:-1]) 
df_feat.head() 


# only bounce rate and exit rate has good correlation

# In[69]:


from sklearn.model_selection import train_test_split 
import scikitplot as skplt
  
X_train, X_test, y_train, y_test = train_test_split( 
      scaled_features, df['Revenue'], test_size = 0.30) 
  
from sklearn.neighbors import KNeighborsClassifier 
  
knn = KNeighborsClassifier(n_neighbors = 17) 
  
knn.fit(X_train, y_train) 
pred = knn.predict(X_test) 

from sklearn.metrics import classification_report, confusion_matrix 
plt_2 = skplt.metrics.plot_confusion_matrix(y_test,pred, normalize=True)
print(classification_report(y_test, pred)) 


# In[ ]:




