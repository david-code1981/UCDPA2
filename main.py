# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import pandas as pd # read CSV file, data processing
import numpy as np # recommendation system
import matplotlib.pyplot as plt # data visualisation
import seaborn as sns # data visualisation


import warnings # to ignore warnings
warnings.filterwarnings("ignore") # no warnings will be printed from now on.

df = pd.read_csv(r"/Users/david/Downloads/UCDPA Project Folder/UCDPA_project_netflix_titles.csv")
# import csv file

# DATA OVERVIEW

print(df.info())
# 8807 total entries, 12 columns
# data type for 'release_year' is an integer.
# data type for 'date_added' is an object, change to DateTime
# can already see 'director', 'cast', 'country' columns have substantial missing values.

print(df.shape)
# There are 8807 entries and 12 columns

print(df.head())
# shows the first 5 rows of the dataset

print(df.tail())
# shows the last 5 rows of the dataset
# column 'type' represents either a Movie or TV Show
# column 'duration' has both minutes and seasons

print(df.columns)
# below is a description of column names:

#show_id: unique id of each show (not much of a use for us in this notebook)
#type: The category of a show, can be either a Movie or a TV Show
#title: Name of the show
#director: Name of the director(s) of the show
#cast: Name of actors and other cast of the show
#country: Name of countries the show is available to watch on Netflix
#date_added: Date when the show was added on Netflix
#release_year: Release year of the show
#rating: Show rating on netflix
#duration: Time duration of the show
#listed_in: Genre of the show
#description: Some text describing the show

print(df.describe())
# oldest is 1925
# youngest is 2021
# mean is 2014

# CLEAN & VALIDATE

netflix_df = df.copy()
# make a copy of dataset

print(netflix_df.info())
# check info of df copy
# 'director', 'cast', 'country' have substantial missing values.

print(netflix_df.duplicated())
# This boolean check is not much use to use as it's only showing top and bottom 5 in this dataset.

print(netflix_df.duplicated().sum())
# there are no duplicates in this dataset.

print(netflix_df.isnull())
# can see there are true values reflecting missing values.

print(netflix_df.isnull().sum())
# too many missing values for 'director', 'cast', 'country' to drop these rows without affecting the dataset quality.
# we will replace these values with 'no data'.

# % of rows missing in each column
for column in netflix_df.columns:
    percentage = netflix_df[column].isnull().mean()
    print(f'{column}: {round(percentage * 100, 2)}%')

# this further shows that percentages for 'dated_added', 'rating', 'duration' are so low we can
# drop these rows without the impacting the integrity of the dataset.

netflix_df['country'] = netflix_df['country'].fillna(netflix_df['country'].mode()[0])
netflix_df['cast'].replace(np.nan,'No data',inplace=True)
netflix_df['director'].replace(np.nan,'No data',inplace=True)

# we will insert code here for 'director', 'cast', 'country' as mentioned above.
# replace these missing values with 'no data'.

print(netflix_df.head())
# check to make sure missing values have been replaced with 'no data'.

print(netflix_df.tail())
# check to make sure missing values have been replaced with 'no data'.

print(netflix_df.isnull().sum())
# we are left with the missing values for 'dated_added', 'rating', 'duration' columns.
# as these are a small number we drop these rows withing reducing the quality of the dataset.

netflix_df.dropna(axis=0, how='any', inplace=True)
# drop all rows with missing values and overwrite the dataset.

print(netflix_df.isnull().sum())
# check to see if there still missing values

print(netflix_df.columns)
# check the no column has been dropped

print(netflix_df.info())
# check the information in our resulting dataset.

print(netflix_df.dtypes)
# check data type of each column.

netflix_df['date_added'] = pd.to_datetime(df['date_added'])
# change 'date_added' column type to DateTime using pandas datetime function.

print(netflix_df.dtypes)
# check data types again

netflix_df = netflix_df.rename(columns={"listed_in":"genre"})
netflix_df['genre'] = netflix_df['genre'].apply(lambda x: x.split(",")[0])
netflix_df['genre'].head()

# rename the 'listed_in' column as 'genre' for easy understanding.

# EXPLORATORY DATA ANALYSIS

# 1) ANALYSIS OF MOVIES VS TV SHOWS

# Does Netflix have more Movies or TV Shows?
# Graph no. 1
sns.set(style="darkgrid")
plt.title("Movie v TV Show")
ax = sns.countplot(x="type", data=netflix_df, palette=('Red','Blue'))

# can see Movies outnumber TV Shows by more than double the amount.

# Percentage of titles that are either Movies or TV Show?
# Graph no. 2

plt.figure(figsize=(12,6))
plt.title("Percentage of Titles that are either Movies or TV Shows")
g = plt.pie(netflix_df.type.value_counts(), explode=(0.025,0.025), labels=netflix_df.type.value_counts().index, colors=['red','blue'],autopct='%1.1f%%', startangle=180);
plt.legend()
plt.show()
# To further illustrate the above we can see Movies represent over two thirds of the titles with 69.7%
# With such a large amount of Movie titles it may be ideal to carry out the majority of the EDA based on Movies.

# 2) RATINGS ANALYSIS

# Count of ratings across titles?
# Graph no. 3
plt.figure(figsize=(12,10))
sns.set(style="darkgrid")
ax = sns.countplot(x="rating", data=netflix_df, palette=("Set3"), order=netflix_df['rating'].value_counts().index[0:15])
plt.title('Count of Titles Across Ratings')
plt.show()
# The largest count of movie titles is under the 'TV-MA' which is a title designed for mature audiences.
# After this the second largest is the 'TV-14' which stands for content inappropriate for children younger than
# 14 years of age.

# Movies vs TV Shows by ratings?
# Graph no. 4
netflix_df.rating.value_counts()
order =  ['G', 'TV-Y', 'TV-G', 'PG', 'TV-Y7', 'TV-Y7-FV', 'TV-PG', 'PG-13', 'TV-14', 'R', 'NC-17', 'TV-MA']
plt.figure(figsize=(17,7))
g = sns.countplot(netflix_df.rating, hue=netflix_df.type, order=order, palette=('Red','Blue'));
plt.title("Ratings for Movies & TV Shows")
plt.xlabel("Rating")
plt.ylabel("Total Count")
plt.show()

# POPULAR GENRE ANALYSIS

# What are the different genres?
count_by_genre = netflix_df['genre'].value_counts()
print(count_by_genre)

# What are the most popular genres?
# Graph no. 5
popular_genres = netflix_df.set_index('title').genre.str.split(', ', expand=True).stack().reset_index(level=1, drop=True);

plt.figure(figsize=(8,10))
g = sns.countplot(y = popular_genres, order=popular_genres.value_counts().index[:20],palette=("Set3"))
plt.title('Top 20 Genres on Netflix')
plt.xlabel('Number of Titles')
plt.ylabel('Genre')
plt.show()

# ANALYSING TITLES THROUGH THE YEARS:

# Availability of content over the years?
# Graph no. 6
netflix_df.plot(kind='scatter', x='date_added', y='release_year', figsize=(17,8), s=4, c='green')
plt.title('Streaming Availability Year vs Actual Release Year')
plt.xlabel('Year Available on Netflix')
plt.ylabel('Year Originally Released')
plt.show()

# ANALYSING MOVIE TITLES BY DURATION:

print(netflix_df['duration'].isnull().sum())
# check to see if there still missing duration values

print(netflix_df['duration'].value_counts())
# check of the count of different durations

movie_df = netflix_df[(netflix_df['type'] == 'Movie')]
print(movie_df.head())
# create a separate Movie dataset as this forms the majority of content.
# we can check the trend of movie durations over time.

print(movie_df['duration'].value_counts())
# can see there are some outliers in Movie duration.
# there is also the string 'min' included in the durations.

movie_df['duration']=movie_df['duration'].str.replace(' min','')
movie_df['duration']=movie_df['duration'].astype(str).astype(int)
print(movie_df['duration'])
# we need to strip the 'min' and replace with an empty string.
# we also set the data type to interger.

# Duration of Movies by Count:
# Graph no. 7
sns.set(style="darkgrid")
sns.kdeplot(data=movie_df['duration'], shade=True)
plt.title('Distribution of Movie Durations')
plt.show()

# now we can create a KDE graph to illustrate the distribution of Movie durations.
# a KDE graph is useful here as we have a large number of data points.

# a large amount of movies are between 75-120 mins. This makes sense as a lot of people find a
# 3 hour movie too long to sit through in one go.

# Trend of Movie Duration by Year:
# Graph no. 8
duration_by_year = movie_df.groupby(['release_year']).mean()
duration_by_year = duration_by_year.sort_index()

plt.figure(figsize=(15,6))
sns.lineplot(x=duration_by_year.index, y=duration_by_year.duration.values)
plt.box(on=None)
plt.ylabel('Movie duration in minutes')
plt.xlabel('Year of release')
plt.title("Movie Duration over the Years", fontsize=14, color='blue')
plt.show()

# can see the change in Movie duration over time.
# 1960 - 1970 has the highest duration in minutes before it levels out in the later part of the century.

# Number of content released by year:
# Graph no. 9
release_year = movie_df['release_year'].value_counts()
release_year = release_year.sort_index(ascending=True)

plt.figure(figsize=(9,8))
plt.plot(release_year[-11:-1])
plt.scatter(release_year[-11:-1].index, release_year[-11:-1].values, s=0.5*release_year[-11:-1].values, c='Red')
plt.box(on=None)
plt.xticks(rotation = 60)
plt.xticks(release_year[-11:-1].index)
plt.title('Number of Content Released by Year', color='blue', fontsize=14)
plt.show()

# INSPECTING IMDB TITLE DATASET/MERGING WITH NETFLIX DATASET

#Reading and Inspecting Movie titles data for IMDb
IMDb_user_ratings = pd.read_csv(r'/Users/david/Downloads/UCDPA Project Folder/IMDB-Ratings.csv')
print(IMDb_user_ratings.head())

# there was a sharp increase in content released from 2014 tp 2018.
# the number of content released by year peaked in 2017 & 2018.

print(IMDb_user_ratings.info())
# as there is only 5 columns and we will be margin this to the netflix dataset we will not drop any columns.

print(IMDb_user_ratings.isnull().sum())

netflix_IMDb_df = pd.merge(netflix_df,IMDb_user_ratings, how='inner', on ='title')
print(netflix_IMDb_df.head())

print(netflix_IMDb_df.isnull().sum())

print(netflix_IMDb_df.shape)

#Sorting in a descending order, so as the output should reflect highest rated movies on top
netflix_IMDb_df.sort_values(by=['averageRating'],inplace=True , ascending = False)
print(netflix_IMDb_df.head(20))

# NETFLIX RECOMMENDATION SYSTEM

#Plot description based Recommender:
#We will calculate similarity scores for all movies based on their plot descriptions and recommend movies based
#on that similarity score. The plot description is given in the description feature of our dataset.

# We need to convert the word vector of each overview.
# We'll compute Term Frequency-Inverse Document Frequency (TF-IDF) vectors for each description.
# The overall importance of each word to the documents in which they appear is equal to TF * IDF.
# This is done to reduce the importance of words that occur frequently in plot overviews and therefore,
# their significance in computing the final similarity score.

from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(stop_words='english')

#Replace missing values with an empty string
netflix_df['description'] = netflix_df['description'].fillna('')

#Create the required TF-IDF matrix by fitting and transforming
# the data
tfidf_matrix = tfidf.fit_transform(netflix_df['description'])

#shape of tfidf_matrix
print(tfidf_matrix.shape)

#Linear Kernel
from sklearn.metrics.pairwise import linear_kernel

#Cosine similarity
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

indices = pd.Series(netflix_df.index, index = netflix_df['title']).drop_duplicates()

print(indices)


def request_recommendation_for(title, cosine_sim=cosine_sim):
    idx = indices[title]

    # Get the pairwise similarity scores of all titles with that title
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort the titles based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the 10 most similar titles
    sim_scores = sim_scores[1:11]

    # Get the title indices
    title_indices = [i[0] for i in sim_scores]

    # return the top 10 similar titles
    return netflix_df['title'].iloc[title_indices]

print(request_recommendation_for('Savages'))
# I have chosen 'savages' from the top 20 list created above.
# capital first letter is required for this model.

print(request_recommendation_for('Narcos'))
# I have chosen 'narcos' as a high profile international Netflix title.
# capital first letter is required for this model.

# We can see the model performs well, but is not very accurate.
# Therefore, more metrics are added to the model to improve performance.
# This is called a content based recommender system

#For this recommender system the content of the movie (cast, description, director,genre etc) is used to find
#its similarity with other movies. Then the movies that are most likely to be similar are recommended.

filledna=netflix_df.fillna('')
print(filledna.head())
# Filling missing values with empty string

# Cleaning the data, making all the words lower case
def clean_data(x):
    return str.lower(x.replace(" ",""))
# this function returns a string with all lower case replaces white spaces.

features=['title', 'director', 'cast', 'genre', 'description']
filledna=filledna[features]
#Features on which the model is to be filtered

for feature in features:
    filledna[feature] = filledna[feature].apply(clean_data)

print(filledna.head())
# takes the features variable created above and applies the clean_data function to it.

# Creating a "soup" or a "bag of words" of all rows
def create_soup(x):
    return x['title']+ ' '+ x['director']+ ' '+ x['cast']+ ' ' + x['genre']+ ' ' +x['description']
# this function concatenates each metric.

filledna['soup'] = filledna.apply(create_soup, axis=1)

# From here on, the code is basically similar to the upper model except the fact that count vectorizer
# is used instead of tfidf

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

count = CountVectorizer(stop_words='english')
count_matrix = count.fit_transform(filledna['soup'])

cosine_sim2 = cosine_similarity(count_matrix, count_matrix)

filledna = filledna.reset_index()
indices = pd.Series(filledna.index, index=filledna['title'])


def get_recommendation_new(title, cosine_sim=cosine_sim):
    idx = indices[title]

    # Get the pairwise similarity scores of all movies with that movie
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort the movies based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the 10 most similar movies
    sim_scores = sim_scores[1:11]

    # Get the movie indices
    movie_indices = [i[0] for i in sim_scores]

    # return the top 10 similar movies
    return netflix_df['title'].iloc[movie_indices]

print(get_recommendation_new('savages', cosine_sim2))
# no capital first letter required for this improved model.


print(get_recommendation_new('narcos', cosine_sim2))
# no capital first letter required for this improved model.


