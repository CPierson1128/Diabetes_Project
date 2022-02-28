#!/usr/bin/env python
# coding: utf-8

# 

# # What impact does race, age, gender and change in certain medications have on average length of stay?

# In[1]:


import numpy as np
import pandas as pd  
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
## Previous analysis using groupby and the differnt columns showed "null" values of ? and Unknown/Invalid
missing_values = ["?", "Unknown/Invalid"]
df = pd.read_csv('https://raw.githubusercontent.com/niteen11/DataAnalyticsAcademy/master/Python/dataset_diabetes/diabetic_data.csv',
                 na_values = missing_values)


# In[2]:


df.shape


# In[3]:


df.columns


# #### Previous Data Analysis Done using groupby to count patient number and each column in the dataset: found null values weight had a large volume of null so I decided to drop. I chose three medications to keep that had a large amount of data: metformin, glipizide, and insulin.

# In[4]:


drop_cols = ['glyburide','weight','payer_code', 'num_lab_procedures', 'num_procedures', 'num_medications','diag_1','diag_2', 'diag_3', 'number_diagnoses', 'max_glu_serum', 'A1Cresult', 'repaglinide', 'nateglinide', 'chlorpropamide',
       'glimepiride', 'acetohexamide', 'tolbutamide','pioglitazone', 'rosiglitazone', 'acarbose', 'miglitol', 'troglitazone',
       'tolazamide', 'examide', 'citoglipton', 'glyburide-metformin', 'glipizide-metformin','glimepiride-pioglitazone', 'metformin-rosiglitazone','metformin-pioglitazone']
df.drop(drop_cols, inplace = True, axis=1)


# In[5]:


df.tail()


# In[6]:


df.isnull().sum()


# In[7]:


## removing rows with no value in race
df_clean = df.dropna(subset=['race'])
df_clean.shape


# In[8]:


## adding column to define length of stay as short (1-3), medium (4-7) and long (8+)
def stay_length(row):
    if row['time_in_hospital'] <= 3:
        return 'short'
    elif row['time_in_hospital'] > 3 and row['time_in_hospital'] <= 7:
        return 'medium'
    else:
        return 'long'

df_clean['Stay_Length'] = df_clean.apply(lambda row: stay_length(row), axis=1)


# In[9]:


df_clean.columns


# #### I only want to keep one patient entry per stay so I am making an assumption that if it is the same patient number and length of stay it is the same stay and kept one entry.

# In[10]:


df_clean.drop_duplicates(subset=['patient_nbr', 'time_in_hospital'], keep='first', inplace=True)


# In[11]:


df_clean.shape


# In[12]:


df_clean.groupby(['gender'])['time_in_hospital'].mean()


# ### There is not a significant difference in average length of stay between men and women

# In[13]:


df_clean.groupby(['gender'])['time_in_hospital'].count()


# ### More women than men were hospitalized 

# In[14]:


df_clean.groupby(['race'])['time_in_hospital'].mean()


# ### Asians had the shortest average stay and African Americans had the highest average stay 

# #### Visualizations by race

# In[15]:


sns.boxplot(x='race',y='time_in_hospital',data=df_clean)


# In[16]:


length_of_stay_by_race = df_clean.groupby('race')['time_in_hospital'].mean()

ax = length_of_stay_by_race.plot(kind='bar', figsize=(10,6), fontsize=13)
ax.set_alpha(0.8)
ax.set_title("Mean days of stay by race")
ax.set_ylabel("Mean Days Stayed", fontsize=15)


# In[17]:


df_clean.groupby(['race'])['time_in_hospital'].count()


# In[18]:


sns.countplot(x="race",data=df_clean,palette="Greens_d")


# ### Caucasians had a significantly greater amount of patients hospitalized 

# In[19]:


df_clean.groupby(['age'])['time_in_hospital'].count()


# In[20]:


sns.boxplot(x='age',y='time_in_hospital',data=df_clean)


# In[21]:


df_clean.groupby(['age'])['time_in_hospital'].mean()


# ### Average length of stay increases with age until it dips slightly after 90 but the largest amount of patients are between 70 - 80.

# In[22]:


grouped_df = df_clean.groupby(['age','race'])['patient_nbr'].count()
print(grouped_df)


# In[23]:


df_clean.groupby(['metformin'])['time_in_hospital'].mean()


# In[24]:


grouped_df = df_clean.groupby(['metformin','Stay_Length'])['patient_nbr'].count()
print(grouped_df)


# In[25]:


sns.boxplot(x='metformin',y='time_in_hospital',hue='race',data=df_clean)
plt.legend(bbox_to_anchor=(1.01, 1),
           borderaxespad=0)


# In[26]:


sns.boxplot(x='metformin',y='time_in_hospital',hue='gender',data=df_clean)
plt.legend(bbox_to_anchor=(1.01, 1),
           borderaxespad=0)


# ### There doesn't seem to be a significant increase or decrease in length of stay when metformin is used 

# In[27]:


df_clean.groupby(['glipizide'])['time_in_hospital'].mean()


# In[28]:


grouped_df = df_clean.groupby(['glipizide','Stay_Length'])['patient_nbr'].count()
print(grouped_df)


# In[29]:


sns.boxplot(x='glipizide',y='time_in_hospital',hue='race',data=df_clean)
plt.legend(bbox_to_anchor=(1.01, 1),
           borderaxespad=0)


# In[30]:


sns.boxplot(x='glipizide',y='time_in_hospital',hue='gender',data=df_clean)
plt.legend(bbox_to_anchor=(1.01, 1),
           borderaxespad=0)


# ### A change in glipizide medication either up or down seems to be correlated with a longer length of stay

# In[31]:


df_clean.groupby(['insulin'])['time_in_hospital'].mean()


# In[32]:


grouped_df = df_clean.groupby(['insulin','Stay_Length'])['patient_nbr'].count()
print(grouped_df)


# In[33]:


sns.boxplot(x='insulin',y='time_in_hospital',hue='race',data=df_clean)
plt.legend(bbox_to_anchor=(1.01, 1),
           borderaxespad=0)


# In[34]:


sns.boxplot(x='insulin',y='time_in_hospital',hue='gender',data=df_clean)
plt.legend(bbox_to_anchor=(1.01, 1),
           borderaxespad=0)


# ### Similar to glipizide it looks like a change in either direction of insulin results in a longer average length of stay 

# ## Next, I looked more closely at each race to see if there were significant differences 

# In[35]:


df_clean_AfricanAmerican = df_clean[df_clean['race'] == 'AfricanAmerican']
df_clean_Asian = df_clean[df_clean['race'] == 'Asian']
df_clean_Caucasian = df_clean[df_clean['race'] == 'Caucasian']
df_clean_Hispanic = df_clean[df_clean['race'] == 'Hispanic']
df_clean_Hispanic.tail()


# In[36]:


df_clean_AfricanAmerican.groupby(['metformin'])['time_in_hospital'].mean()


# In[37]:


df_clean_Asian.groupby(['metformin'])['time_in_hospital'].mean()


# In[38]:


df_clean_Caucasian.groupby(['metformin'])['time_in_hospital'].mean()


# In[39]:


df_clean_Hispanic.groupby(['metformin'])['time_in_hospital'].mean()


# #### Decreasing or maintaining metformin seems shorten length of stay for Hispanics which is different than our overall dataset

# In[40]:


df_clean_AfricanAmerican.groupby(['glipizide'])['time_in_hospital'].mean()


# In[41]:


df_clean_Asian.groupby(['glipizide'])['time_in_hospital'].mean()


# In[42]:


df_clean_Caucasian.groupby(['glipizide'])['time_in_hospital'].mean()


# In[43]:


df_clean_Hispanic.groupby(['glipizide'])['time_in_hospital'].mean()


# In[44]:


df_clean_AfricanAmerican.groupby(['insulin'])['time_in_hospital'].mean()


# In[45]:


df_clean_Asian.groupby(['insulin'])['time_in_hospital'].mean()


# In[46]:


df_clean_Caucasian.groupby(['insulin'])['time_in_hospital'].mean()


# In[47]:


df_clean_Hispanic.groupby(['insulin'])['time_in_hospital'].mean()


# #### I didn't see any other signficantly different findings 

# In[ ]:




