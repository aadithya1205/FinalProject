#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
df=pd.read_csv("influencer.csv")
df


# In[2]:


df.isna().sum()


# In[3]:


df["Country Or Region"].mode()


# In[4]:


df["Country Or Region"]=df["Country Or Region"].fillna("United States")


# In[5]:


df.isna().sum()


# In[6]:


df["Channel Info"]=df["Channel Info"].str.lstrip("\n")


# In[7]:


df


# In[8]:


#CONVERTING m,b,k to million,billion and thousand
def converter(x):
    if 'm' in x:
      
        t1=x.replace("m","")
        a=pd.to_numeric(t1)
        return a*1000000
        
    elif 'b' in x:
        t1=x.replace("b","")
        a=pd.to_numeric(t1)
        return a*1000000000
    elif 'k' in x:
        
        t1=x.replace("k","")
        a=pd.to_numeric(t1)
        return a*1000
    else:
        return x


# In[9]:


df.shape


# In[10]:


df["Followers"]=df["Followers"].astype('str')
df["Avg. Likes"]=df["Avg. Likes"].astype('str')
df["Posts"]=df["Posts"].astype('str')
df["New Post Avg. Likes"]=df["New Post Avg. Likes"].astype('str')
df["Total Likes"]=df["Total Likes"].astype('str')


# In[11]:


df.dtypes


# In[12]:


#Applying the function
df["Followers"]=df["Followers"].apply(converter)
df["Avg. Likes"]=df["Avg. Likes"].apply(converter)
df["Posts"]=df["Posts"].apply(converter)
df["New Post Avg. Likes"]=df["New Post Avg. Likes"].apply(converter)
df["Total Likes"]=df["Total Likes"].apply(converter)


# In[13]:


df.dtypes


# QUESTION1:
# Are there any correlated features in the given dataset? If yes, state the correlation
# coefficient of the pair of features which are highly correlated.

# In[14]:


x_train=df.drop(["Country Or Region","Channel Info"],axis=1)
corr_matrix = x_train.corr()
# Take absolute values of correlated coefficients
corr_matrix = corr_matrix.abs().unstack()
corr_matrix = corr_matrix.sort_values(ascending=False)
# Take only features with correlation above threshold of 0.8
corr_matrix = corr_matrix[corr_matrix >= 0.5]
corr_matrix = corr_matrix[corr_matrix < 1]
corr_matrix = pd.DataFrame(corr_matrix).reset_index()
corr_matrix.columns = ['feature1', 'feature2', 'Correlation']
corr_matrix


# ANS:New Post Avg. Likes AND Avg. Likes HAS CORRELATION OF 0.892784

# QUESTION2:What is the frequency distribution of the following features?
# ○ Influence Score
# ○ Followers
# ○ Posts

# In[16]:


#FOLLOWERS
import matplotlib.pyplot as plt
plt.hist(df["Followers"])


# In[17]:


#INFLUENCE SCORE
plt.hist(df["Influence Score"])


# In[18]:


df["Posts"]=pd.to_numeric(df["Posts"])


# In[19]:


#POSTS
plt.hist(df["Posts"])


# QUESTION3:Which country houses the highest number of Instagram Influencers? Please show the
# count of Instagram influencers in different countries using barchart.

# In[20]:


fig = plt.figure(figsize =(30, 10))
plt.bar(df["Country Or Region"].unique(),df["Country Or Region"].value_counts(),color ='maroon',width = 1)

plt.xlabel("Country")
plt.ylabel("Influencers count")
plt.title("Number of influencers in each country")
plt.show()



# ANS:Spain has the highest number of influencers

# QUESTION5:Who are the top 10 influencers in the given dataset based on the following features
# ● Followers
# ● Average likes
# ● Total Likes

# In[66]:


df2= df.sort_values(by=['Followers'],ascending=False).head(10)


# Top 10 Influencers based on Followers

# In[21]:


fig = plt.figure(figsize =(30, 10))
plt.bar(df2["Channel Info"],df2["Followers"],color ='maroon',width = 0.5)

plt.xlabel("Country")
plt.ylabel("Influencers")
plt.title("Top 10 Influencers based on Followers")
plt.show()


# In[22]:


df3= df.sort_values(by=['Avg. Likes'],ascending=False).head(10)
df3


# Top 10 Influencers based on Average Likes

# In[23]:


fig = plt.figure(figsize =(30, 10))
plt.bar(df3["Channel Info"],df3["Avg. Likes"],color ='maroon',width = 0.5)

plt.xlabel("Country")
plt.ylabel("Influencers")
plt.title("Top 10 Influencers based on Average Likes")
plt.show()


# In[24]:


df4= df.sort_values(by=['Total Likes'],ascending=False).head(10)
df4


# Top 10 Influencers in based on Total Likes

# In[25]:


fig = plt.figure(figsize =(30, 10))
plt.bar(df4["Channel Info"],df4["Total Likes"],color ='maroon',width = 0.5)

plt.xlabel("Country")
plt.ylabel("Influencers")
plt.title("Top 10 Influencers in based on Total Likes")
plt.show()


# QUESTION5:Describe the relationship between the following pairs of features using a suitable graph
# ● Followers and Total Likes
# ● Followers and Influence Score
# ● Posts and Average likes
# ● Posts and Influence Score

# In[26]:


x=df["Followers"]
y=df["Total Likes"]
plt.plot(x,y)
plt.title("Followers and Total Likes")
plt.show()


# In[27]:


x=df["Followers"]
y=df["Influence Score"]
plt.plot(x,y)
plt.title("Followers and Influence score")
plt.show()


# In[28]:


x=df["Posts"]
y=df["Avg. Likes"]
plt.scatter(y,x)
plt.title("Average likes and Posts")
plt.show()


# In[29]:


x=df["Posts"]
y=df["Influence Score"]
plt.scatter(y,x)
plt.title("Influence score and posts")
plt.show()


# In[ ]:




