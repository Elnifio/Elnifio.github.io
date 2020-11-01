#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Import Libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import csr_matrix
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error
import itertools
from sklearn.metrics import silhouette_samples, silhouette_score

get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


# Import the Movies dataset
movies = pd.read_csv('Desktop/2020 Fall/stor565/final project/archive/movies.csv') 
movies.shape


# In[3]:


# Import the ratings dataset
ratings = pd.read_csv('Desktop/2020 Fall/stor565/final project/archive/ratings.csv') 


# In[4]:


print('The dataset contains: ', len(ratings), ' ratings of ', len(movies), ' movies.')


# In[5]:


# Function to get the genre ratings
def get_genre_ratings(ratings, movies, genres, column_names): 
   genre_ratings = pd.DataFrame()
   for genre in genres:
       genre_movies = movies[movies['genres'].str.contains(genre) ]
       avg_genre_votes_per_user = ratings[ratings['movieId'].isin(genre_movies['movieId'])].loc[:, ['userId', 'rating']].groupby(['userId'])['rating'].mean().round(2)
       genre_ratings = pd.concat([genre_ratings, avg_genre_votes_per_user], axis=1)
   genre_ratings.columns = column_names 
   return genre_ratings

genre_ratings = get_genre_ratings(ratings, movies, ['Romance', 'Sci-Fi'], ['avg_romance_rating', 'avg_scifi_rating']) 
genre_ratings.head()


# In[6]:


# Function to get the biased dataset
def bias_genre_rating_dataset(genre_ratings, score_limit_1, score_limit_2):
   biased_dataset =    genre_ratings[((genre_ratings['avg_romance_rating'] < score_limit_1 - 0.2) & (genre_ratings['avg_scifi_rating'] > score_limit_2)) | ((genre_ratings['avg_scifi_rating'] < score_limit_1) & (genre_ratings['avg_romance_rating'] > score_limit_2))]
   #biased_dataset = pd.concat([biased_dataset[:300], genre_ratings[:2]])
   biased_dataset = biased_dataset[:300]
   biased_dataset = pd.DataFrame(biased_dataset.to_records())
   return biased_dataset
# Bias the dataset
biased_dataset = bias_genre_rating_dataset(genre_ratings, 3.2, 2.5)
# Printing the resulting number of records & the head of the dataset
print( "Number of records: ", len(biased_dataset))
biased_dataset.head()
   


# In[7]:


# Defining the scatterplot drawing function
def draw_scatterplot(x_data, x_label, y_data, y_label):
   fig = plt.figure(figsize=(8,8))
   ax = fig.add_subplot(111)
   plt.xlim(0, 6)
   plt.ylim(0, 6) 
   ax.set_xlabel(x_label) 
   ax.set_ylabel(y_label) 
   ax.scatter(x_data, y_data, s=30)
# Plot the scatterplot
draw_scatterplot(biased_dataset['avg_scifi_rating'],'Avg scifi rating', biased_dataset['avg_romance_rating'], 'Avg romance rating')


# In[8]:



# Let's turn our dataset into a list
X = biased_dataset[['avg_scifi_rating','avg_romance_rating']].values
# Import KMeans
from sklearn.cluster import KMeans
# Create an instance of KMeans to find two clusters
kmeans_1 = KMeans(n_clusters=2)
# Use fit_predict to cluster the dataset
predictions = kmeans_1.fit_predict(X)


# In[9]:



# Defining the cluster plotting function
def draw_clusters(biased_dataset, predictions, cmap='viridis'):
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111)
    plt.xlim(0, 5)
    plt.ylim(0, 5)
    ax.set_xlabel('Avg scifi rating')
    ax.set_ylabel('Avg romance rating')
    clustered = pd.concat([biased_dataset.reset_index(), pd.DataFrame({'group':predictions})], axis=1)
    plt.scatter(clustered['avg_scifi_rating'], clustered['avg_romance_rating'], c=clustered['group'], s=20, cmap=cmap)
# Plot
draw_clusters(biased_dataset, predictions)


# In[10]:


# Selecting our dataset to study
df = biased_dataset[['avg_scifi_rating','avg_romance_rating']]
# Choose the range of k values to test.
# We added a stride of 5 to improve performance. We don't need to calculate the error for every k value
possible_k_values = range(2, len(X)+1, 5)
# Define function to calculate the clustering errors
def clustering_errors(k, data):
    kmeans = KMeans(n_clusters=k).fit(data)
    predictions = kmeans.predict(data)
    #cluster_centers = kmeans.cluster_centers_
    # errors = [mean_squared_error(row, cluster_centers[cluster]) for row, cluster in zip(data.values, predictions)]
    # return sum(errors)
    silhouette_avg = silhouette_score(data, predictions)
    return silhouette_avg


# In[11]:


# Calculate error values for all k values we're interested in
errors_per_k = [clustering_errors(k, X) for k in possible_k_values]
# Plot the each value of K vs. the silhouette score at that value
fig, ax = plt.subplots(figsize=(16, 6))
plt.plot(possible_k_values, errors_per_k)
# Ticks and grid
xticks = np.arange(min(possible_k_values), max(possible_k_values)+1, 5.0)
ax.set_xticks(xticks, minor=False)
ax.set_xticks(xticks, minor=True)
ax.xaxis.grid(True, which='both')
yticks = np.arange(round(min(errors_per_k), 2), max(errors_per_k), .05)
ax.set_yticks(yticks, minor=False)
ax.set_yticks(yticks, minor=True)
ax.yaxis.grid(True, which='both')


# In[12]:


#Looking at the plot, we can see that the best choices of the K values are: 7, 17, 22


# In[13]:


# Create an instance of KMeans to find seven clusters
kmeans_4 = KMeans(n_clusters=17)
# Use fit_predict to cluster the dataset
predictions_4 = kmeans_4.fit_predict(X)
# Plot
draw_clusters(biased_dataset, predictions_4, cmap='Accent')


# In[14]:


#adding action to moive genre 
# Select our biased dataset and add action genre
biased_dataset_3_genres = get_genre_ratings(ratings, movies, ['Romance','Sci-Fi', 'Action'],                                          
['avg_romance_rating', 'avg_scifi_rating', 'avg_action_rating'])

# Drop null values
biased_dataset_3_genres = bias_genre_rating_dataset(biased_dataset_3_genres, 3.2, 2.5).dropna()

# Print the number of records and the head of our dataset
print( "Number of records: ", len(biased_dataset_3_genres))
biased_dataset_3_genres.head()


# In[15]:


#higher level clustering 
# Merge the two tables then pivot so we have Users X Movies dataframe
ratings_title = pd.merge(ratings, movies[['movieId', 'title']], on='movieId' )
len(ratings_title.userId.unique())
len(ratings_title.title.unique())


# In[16]:


user_movie_ratings = pd.pivot_table(ratings_title.head(1000000), index='userId', columns= 'title', values='rating')


# In[17]:


print('dataset dimensions: ', user_movie_ratings.shape, '\n\nSubset example:')
user_movie_ratings.iloc[:6, :10]


# In[18]:


# Define Function to get the most rated movies
def get_most_rated_movies(user_movie_ratings, max_number_of_movies):
    user_movie_ratings = user_movie_ratings.append(user_movie_ratings.count(), ignore_index=True)
    user_movie_ratings_sorted = user_movie_ratings.sort_values(len(user_movie_ratings)-1, axis=1, ascending=False)
    user_movie_ratings_sorted = user_movie_ratings_sorted.drop(user_movie_ratings_sorted.tail(1).index)
    most_rated_movies = user_movie_ratings_sorted.iloc[:, :max_number_of_movies]
    return most_rated_movies
                                                               
def get_users_who_rate_the_most(most_rated_movies, max_number_of_movies):
    # Get most voting users
    # 1- Count
    most_rated_movies['counts'] = pd.Series(most_rated_movies.count(axis=1))
    # 2- Sort
    most_rated_movies_users = most_rated_movies.sort_values('counts', ascending=False)
    # 3- Slice
    most_rated_movies_users_selection = most_rated_movies_users.iloc[:max_number_of_movies, :]
    most_rated_movies_users_selection = most_rated_movies_users_selection.drop(['counts'], axis=1)
    
    return most_rated_movies_users_selection
    


# In[19]:


# Define the sorting by rating function
def sort_by_rating_density(user_movie_ratings, n_movies, n_users):
    most_rated_movies = get_most_rated_movies(user_movie_ratings, n_movies)
    most_rated_movies = get_users_who_rate_the_most(most_rated_movies, n_users)
    return most_rated_movies

# choose the number of movies and users and sort
n_movies = 33000
n_users = 40000
most_rated_movies_users_selection = sort_by_rating_density(user_movie_ratings, n_movies, n_users)

most_rated_movies_users_selection.head()


# In[20]:


def draw_movies_heatmap(most_rated_movies_users_selection, axis_labels=True):
    
    # Reverse to match the order of the printed dataframe
    #most_rated_movies_users_selection = most_rated_movies_users_selection.iloc[::-1]
    
    fig = plt.figure(figsize=(15,4))
    ax = plt.gca()
    
    # Draw heatmap
    heatmap = ax.imshow(most_rated_movies_users_selection,  interpolation='nearest', vmin=0, vmax=5, aspect='auto')

    if axis_labels:
        ax.set_yticks(np.arange(most_rated_movies_users_selection.shape[0]) , minor=False)
        ax.set_xticks(np.arange(most_rated_movies_users_selection.shape[1]) , minor=False)
        ax.invert_yaxis()
        ax.xaxis.tick_top()
        labels = most_rated_movies_users_selection.columns.str[:40]
        ax.set_xticklabels(labels, minor=False)
        ax.set_yticklabels(most_rated_movies_users_selection.index, minor=False)
        plt.setp(ax.get_xticklabels(), rotation=90)
    else:
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    
    ax.grid(False)
    ax.set_ylabel('User id')

    # Separate heatmap from color bar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)

    # Color bar
    cbar = fig.colorbar(heatmap, ticks=[5, 4, 3, 2, 1, 0], cax=cax)
    cbar.ax.set_yticklabels(['5 stars', '4 stars','3 stars','2 stars','1 stars','0 stars'])



    plt.show()
    


# In[21]:


draw_movies_heatmap(most_rated_movies_users_selection)


# In[22]:


most_rated_movies_33k = get_most_rated_movies(user_movie_ratings,33000)
most_rated_movies_33k


# In[23]:


sparse_ratings = csr_matrix(most_rated_movies_33k.astype(pd.SparseDtype("float", np.nan)).sparse.to_coo())


# In[24]:


predictions = KMeans(n_clusters=22, algorithm='full').fit_predict(sparse_ratings)


# In[74]:


clustered = pd.concat([most_rated_movies_33k.reset_index(), pd.DataFrame({'group':predictions})], axis=1)


# In[121]:


for i in range(1,23):
    cluster_number = 3

    # Let's filter to only see the region of the dataset with the most number of values 
    n_users = 10000
    n_movies = 30000
    cluster = clustered[clustered.group == cluster_number].drop(['index', 'group'], axis=1)

    cluster = sort_by_rating_density(cluster, n_movies, n_users)

    for j in cluster.index:
        user_id = j

        # Get all this user's ratings
        user_2_ratings  = cluster.loc[user_id, :]

        # Which movies did they not rate? (We don't want to recommend movies they've already rated)
        user_2_unrated_movies =  user_2_ratings[user_2_ratings.isnull()]

        # What are the ratings of these movies the user did not rate?
        avg_ratings = pd.concat([user_2_unrated_movies, cluster.mean()], axis=1, join='inner').loc[:,0]

        # Let's sort by rating so the highest rated movies are presented first
        print(avg_ratings[:5])


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




