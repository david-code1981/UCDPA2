# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load
import matplotlib_inline
import pandas as pd # read CSV file, data processing
import numpy as np # recommender system
import regex as re # regex
import matplotlib.pyplot as plt # data visualisation
import seaborn as sns # data visualisation

import warnings
warnings.filterwarnings("ignore") # no warnings will be printed from now on.

# import csv file
df = pd.read_csv(r"/Users/david/Downloads/UCDPA Project Folder/UCDPA_project_netflix_titles.csv")

# DATA OVERVIEW

print(df.info())
# 8807 total entries, 12 columns
# 'release_year' is an integer.
# 'date_added' is an object, change to DateTime
# can already see 'director', 'cast', 'country' have substantial missing values.

print(df.shape)
# There are 8807 entries and 12 columns

print(df.head())
# shows the first 5 rows of the dataset

print(df.tail())
# shows the last 5 rows of the dataset
# Colomn'type' is either Movie or TV Show
# Coloumn 'duration' has both minutes and seasons

print(df.columns)
# This lists the column names


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
# This boolean check is not mush use to use as it's only showing top and bottom 5 in this dataset.

print(netflix_df.duplicated().sum())
# there are no duplicates in this dataset.

print(netflix_df.isnull())
# can see there are True values reflecting missing values.

print(netflix_df.isnull().sum())
# too many missing values for 'director', 'cast', 'country' to drop these rows
# replace these values with 'no data' & 'United States' for the 'country'
# 'dated_added', 'rating', 'duration' missing rows can be dropped.

# % of rows missing in each column
for column in netflix_df.columns:
    percentage = netflix_df[column].isnull().mean()
    print(f'{column}: {round(percentage * 100, 2)}%')

# this further shows that percentages for 'dated_added', 'rating', 'duration' so so low we can
# drop these rows without the impacting the integrity of the dataset.

# insert code here for 'director', 'cast', 'country'
# replace these missing values with 'no data' & & 'United States' for the 'country'
netflix_df['country'] = netflix_df['country'].fillna(netflix_df['country'].mode()[0])
netflix_df['cast'].replace(np.nan,'No data',inplace=True)
netflix_df['director'].replace(np.nan,'No data',inplace=True)

print(netflix_df.head())
# check to make sure missing values have been replaced.

print(netflix_df.tail())
# # check to make sure missing values have been replaced.

print(netflix_df.isnull().sum())
# shows we just left with the missing values for 'dated_added', 'rating', 'duration'

# insert code here for 'dated_added', 'rating', 'duration'
# drop the rows for these nan values
netflix_df.dropna(axis=0, how='any', inplace=True)

print(netflix_df.isnull().sum())
# check to see if there still missing values

print(netflix_df.columns)
# check the no column has been dropped

print(netflix_df.info())
# check the information in our resulting dataset.

print(netflix_df.dtypes)
# check data type

netflix_df['date_added'] = pd.to_datetime(df['date_added'])
# change 'date_added' column type to DateTime

print(netflix_df.dtypes)
# check data type

# EXPLORATORY DATA ANALYSIS

# 1) ANALYSIS OF MOVIES VS TV SHOWS

# Does Netflix have more Movies or TV Shows?

# Graph no. 1
sns.set(style="darkgrid")
plt.title("Movie v TV Show")
ax = sns.countplot(x="type", data=netflix_df, palette=('Red','Blue'))
# can see Movies outnumber TV Shows by a large amount

# Percentage of titles that are either Movies or TV Show?
# Graph no. 2

plt.figure(figsize=(12,6))
plt.title("Percentage of Titles that are either Movies or TV Shows")
g = plt.pie(netflix_df.type.value_counts(), explode=(0.025,0.025), labels=netflix_df.type.value_counts().index, colors=['red','blue'],autopct='%1.1f%%', startangle=180);
plt.legend()
plt.show()
# To further illustrate the above we can see Movies represent over two thirds of the titles with 69.7%
# With such a large amount of Movie titles it may be ideal to carry out the majority of the EDA based on Movies.

# 2) MOVIE RATINGS ANALYSIS

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
count_by_genre = netflix_df['listed_in'].value_counts()
print(count_by_genre)

# What are the most popular genres?
# Graph no. 5
popular_genres = netflix_df.set_index('title').listed_in.str.split(', ', expand=True).stack().reset_index(level=1, drop=True);

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
#Graph no. 7
sns.set(style="darkgrid")
sns.kdeplot(data=movie_df['duration'], shade=True)
plt.show()

# now we can create KDE graph to illustrate the duration of Movies by number.
# KDE graph is useful here as we have a large number of data points.

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

# INSECTING IMDB TITLE DATASET/MERGING WITH NETFLIX DATASET

#Reading and Inspecting Movie titles data for IMDb
IMDb_user_ratings = pd.read_csv(r'/Users/david/Downloads/UCDPA Project Folder/IMDB-Ratings.csv')
print(IMDb_user_ratings.head())

# there was a sharp increase in content released from 2014 tp 2018.
# the number of content released by year peaked in 2017 & 2018.

print(IMDb_user_ratings.info())
# as there is only 5 columns and we will be mergin this to the netflix dataset we will not drop any columns.

print(IMDb_user_ratings.isnull().sum())

netflix_IMDb_df = pd.merge(netflix_df,IMDb_user_ratings, how='inner', on ='title')
print(netflix_IMDb_df.head())

print(netflix_IMDb_df.isnull().sum())

print(netflix_IMDb_df.shape)

#Sorting in a descending order, so as the output should reflect highest rated movies on top
netflix_IMDb_df.sort_values(by=['averageRating'],inplace=True , ascending = False)
print(netflix_IMDb_df.head(20))