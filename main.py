# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

import pandas as pd # read CSV file, data processing
import numpy as np # linear algebra??
import regex as re # regex
import matplotlib.pyplot as plt # data visualisation
import seaborn as sns # data visualisation

# import csv file
df = pd.read_csv(r"/Users/david/Downloads/UCDPA Project Folder/UCDPA_project_netflix_titles.csv")

# DATA OVERVIEW

print(df.info())

# 8807 total entries, 12 columns
# 'release_year' is an integer
# 'date_added' is an object, change to DateTime
# can already see 'director', 'cast', 'country' have substantial nan values.

print(df.shape)

# 8807 entries and 12 columns

print(df.head())

# shows the first 5 rows

print(df.tail())

# shoes the last 5 rows

# 'type' is either Movie or TV Show
#'duration' has both minutes and seasons

print(df.columns)

# lists column names

#type is object

#description of column names:
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

#CLEAN AND VALIDATE

netflix_df = df.copy()

# make a copy of dataset

netflix_df.info()

# 'director', 'cast', 'country' have substantial nan values.

print(netflix_df.duplicated())

print(netflix_df.duplicated().sum())

# there are no duplicates

print(netflix_df.isnull())

# can see there are True values

print(netflix_df.isnull().sum())

# too many nan values for 'director', 'cast', 'country' to drop these rows
# replace these values with 'no data'

# nan values in date_added is to be substituted in with the most recent date from date_added.
# This is because Netflix has the tendency to add more content over time, so this would minimally skew analysis
# results.

# 'dated_added', 'rating', 'duration' nan rows can be dropped.
