#!/usr/bin/env python
# coding: utf-8

# # Netflix Movie Recommendation system using NLP
# 
# # Problem
# 
# I am someone who like to watch movies a lot and I can indecisive about the Netflix movie to watch and it can be a really headache . 
# 
# # Goal
# 
# To build a recommendation system that could be guide me on some movies based on my previous watch.
# 
# # Overview
# 
# A Recommendation system that matches the content on the bases of similarities between a given set of users and a set of items(In this case Netflix movies). 
# 
# YouTube uses the recommendation system at a large scale to suggest you videos based on your history. For example, if you watch a lot of data science videos, it would suggest those types of videos. Let us understand this concept more deeply:
# 
# ![](1_3m0Jmc_k0NP3_CCwnwdB7Q.png)
# 
# **Content-based recommenders:** suggest similar items based on a particular item. This system uses item metadata, such as genre, director, description, actors, etc. for movies, to make these recommendations. The general idea behind these recommender systems is that if a person likes a particular item, he or she will also like an item that is similar to it. And to recommend that, it will make use of the user's past item metadata. A good example could be YouTube, where based on your history, it suggests you new videos that you could potentially watch.
# 
# **Collaborative filtering engines:** This type of filter is based on users’ rates, and it will recommend movies that we haven’t watched yet, but users similar to us have and like. To determine whether two users are similar or not, this filter considers the movies both of them watched and how they rated them. Collaborative filters do not require item metadata like its content-based counterparts.
# 
# ***In this approach, we are going to use Content-based filtering since the data we are using does not involve any user information.***
# 
# The data is gathered by Kaggle, which consist of movie and tv series data having 8807 rows and12 columns, Link of the dataset is: [https://www.kaggle.com/shivamb/netflix-shows](https://www.kaggle.com/shivamb/netflix-shows)
# 
# ##### IDE that I am using here is Jupyter notebook

# In[32]:


# Import packages
from unicodedata import category
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import collections

from zmq import TCP_KEEPALIVE_CNT
warnings.filterwarnings('ignore')
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# # Overview of the dataset:

# In[33]:


#load movie data
data = pd.read_csv('netflix_titles.csv')
data.head()
data.shape


# # Filter movie data from the dataset

# In[34]:


#filter movie data from dataset and covert movie title to lowercase
net_movie = data.loc[data.type == 'Movie', :].reset_index()
net_movie.title = net_movie.title.str.lower()
net_movie['index'] = net_movie.index
net_movie.head()


# # Fliter tv shows data from the dataset

# In[35]:


#fliter tv shows data from the data set
tv_shows = data.loc[data.type == 'TV Show', :].reset_index()
tv_shows.title = tv_shows.title.str.lower()
tv_shows['index'] = tv_shows.index
tv_shows.head()


# # Check the duplicate values

# In[36]:


net_movie.duplicated().sum()
tv_shows.duplicated().sum()


# Analysis
# # Comparing number of shows and movies

# In[37]:


#getting index of tv shows
index = tv_shows.index
no_of_rows_tv = len(index)

#getting index of movies
index = net_movie.index
no_of_rows_movies = len(index)


# In[38]:


#comparing the number of tv shows and movies
plt.figure(figsize = (13, 6))
labels = ['TV Shows', 'Movie']
plt.pie(data['type'].value_counts().sort_values(),labels = labels, colors = ['lightblue', 'lightslategrey'], autopct ='%2.2f%%', startangle = 90)
plt.legend(labels = ['TV Shows', 'Movie'], loc = 'upper right')
plt.show()


# # List of latest 15 movies

# In[39]:


top_15 = net_movie.sort_values(by = 'release_year' , ascending=False).head(15)
top_15[['title' , 'release_year']]


# # Country with the most number of titles

# In[40]:


plt.figure(figsize=(15,7))
sns.countplot(data["country"],hue = data["type"],order = data["country"].value_counts().index[:10] , palette = 'deep')


# # Catagory with maximum content

# In[41]:


# Movie category with maximum content
newdata= net_movie
new= newdata.groupby('listed_in').count()
category = new.sort_values(by = 'index' ,ascending=False).head(10)
category1= category[['type']]
category1.plot(kind ="barh")


# In[42]:


#Tv series with maximum content
newtv= tv_shows
new1= newtv.groupby('listed_in').count()
tvcat = new1.sort_values(by = 'index' ,ascending=False).head(10)
tvcat1= tvcat[['type']]
tvcat1.plot(kind ="barh")


# # Duration of top 20 movies with respect to their countries

# In[43]:


# Duration of top 10 movies with respect to their countries
net_movie['time'] = net_movie['duration'].str.strip('min')
net_movie['time'] = net_movie['time'].fillna(0)    #fill empty values
net_movie['time'] = net_movie['time'].astype('int')
net_movie['screenplay'] = net_movie['time']/60

top_20 = net_movie.sort_values(by = 'screenplay' , ascending=False).head(20)
plt.figure(figsize=(12,10))
sns.barplot(data= top_20 , y= 'title' , x='screenplay' , hue = 'country' , dodge = False)
plt.legend(loc = 'lower right')
plt.xlabel('Total hours')
plt.ylabel('Movie')
plt.title('Top 20 Movies by Run Time')
plt.show()


# # Building Recommendation system
# Using cosine similarity scores for all movies based on their plot descriptions and recommend movies based on that similarity score threshold.

# In[44]:


#get column names
net_movie.columns


# # Select a few features and create a column in a data frame that combines all the selected features into one string:

# In[45]:


#selecting features
features = ['director' , 'cast' ,'country' , 'description' ,'listed_in' ]


# # By using the apply method we will transform this function to all the rows of the data frame:

# In[46]:


#create a column in dataframe which combines all the selected features
for feature in features:
    net_movie[features]= net_movie[features].fillna('')
def combine_features(row):
    return row['director'] + ' ' + row['cast'] + ' ' + row['country'] + ' ' + row['description'] + ' ' + row['listed_in']
net_movie['combine_features'] = net_movie.apply(combine_features, axis=1)
print("Combine Feature: " , net_movie['combine_features'])


# #### Scikit-learn's CountVectorizer is used to convert a collection of text documents to a vector of term/token counts.
# Now we need to check the similarities between the content for which we create the count metric and then get the cosine similarity to get the similarity score:

# In[47]:


#create count matrix from this new combine column
cv = CountVectorizer()
count_matrix = cv.fit_transform(net_movie['combine_features'])
cosine_sim = cosine_similarity(count_matrix)


# #### Print the topmost similarity score:

# In[48]:


#get index of the movie from the title
def title_from_index(index):
    return net_movie[net_movie.index == index]['title'].values[0]

def title_from_index(df ,index):
    return df[df.index == index]['title'].values[0]

def title_from_index(df ,title):
    return df[df.title == title]['index'].values[0]


# In[51]:


#get the list of similar movies in descending order of similarity score
def get_recommendations(title, cosine_sim=cosine_sim):
    if title.lower() in [i for i in net_movie['title'].str.lower()]:
        movie_index = title_from_index(net_movie , title)
        similar_movies = list(enumerate(cosine_sim[movie_index]))
        similar_movies = sorted(similar_movies, key=lambda x: x[1], reverse=True)
        similar_movies = similar_movies[1:6]
        movie_indices = [i[0] for i in similar_movies]
        mov_list = net_movie['title'].iloc[movie_indices]
        print('Recommended Movies for  \' {} \' in descending order'.format(title))
        print('-'*(48+len(title)))
        print(*mov_list, sep = "\n")

    else:
        print('We have no movie to recommend for \' {} \' at that time. Try with other similar movies.'.format(title))
    
get_recommendations('the matrix')


# #### Great! I have selected a movie called “the matrix” and on the bases of similarities between the movie top 5 similar movies are listed.
