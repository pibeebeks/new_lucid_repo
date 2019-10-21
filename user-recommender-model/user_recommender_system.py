'''Function that gets similar users based on bios of users'''
# -*- coding: utf-8 -*-

#Importing utilities

import pickle
import pandas as pd
from sqlalchemy import create_engine
from sklearn.metrics.pairwise import linear_kernel
from sklearn.feature_extraction.text import TfidfVectorizer
import mysql.connector






#Getting the data from the database and showing the tables

MYDB = mysql.connector.connect(host="remotemysql.com",
                               user="8SawWhnha4",
                               passwd="zFvOBIqbIz",
                               database="8SawWhnha4")

DBCURSOR = MYDB.cursor()
DBCURSOR.execute("show tables")
for table in DBCURSOR:
    print(table)


#Extracting the table relevant for users and their bio



ENGINE = create_engine('mysql+mysqlconnector://8SawWhnha4:zFvOBIqbIz@remotemysql.com/8SawWhnha4')
# Looking at the Users table

USERS = pd.read_sql_query("select * from users", ENGINE)



# Dropping columns considered irrelevant
USERS = USERS.drop(["username", "email", "image", "provider", "provider_id", "password",
                    "remember_token", "created_at", "updated_at"], axis=1)
# Checking the outcome


#Checking other tables showed that id here means user_id
#So we rename the id clumn to userid

USERS.rename(columns={"id":"user_id"}, inplace=True)
INDICIES = pd.Series(USERS['user_id'].index)



# filling the missing values with empty string and dropping empty short_bio rows
USERS['short_bio'] = USERS['short_bio'].fillna('')
USERS.drop(USERS[USERS['short_bio'] == ''].index, inplace=True)

#Taking out special characters
USERS['short_bio'] = USERS['short_bio'].str.replace(r'\W', ' ')

#Checking


# Importing

USERS_TFIDF = TfidfVectorizer(stop_words='english')

# computing TF-IDF matrix required for calculating cosine similarity
USERS_TRANSF = USERS_TFIDF.fit_transform(USERS['short_bio'])

COSINE_SIMILARITY = linear_kernel(USERS_TRANSF, USERS_TRANSF)

#Declaring a function that uses the model to fetch similar users based on user_bio

def recommend(index, cosine_sim=COSINE_SIMILARITY):
    '''
    Function to who to follow similar to user based on short bio

    Parameters
    index
    (int):user id
    cosine_sim
    (int):cosine similarity of all users

    Returns
    (list): 10 most popular users similar to user or
    'This user has no follower recommemdation' when no user has close cosine similarity
    '''
    try:
        i_d = INDICIES[index]
        # Get the pairwsie similarity scores of all names
        # sorting them and getting top 10
        similarity_scores = list(enumerate(cosine_sim[i_d]))
        similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
        similarity_scores = similarity_scores[1:11]

        # Get the names index
        users_index = [i[0] for i in similarity_scores]

        # Return the top 10 most similar names
        return USERS['name'].iloc[users_index]
    except KeyError:
        return 'This user has no bio description/no article recommendation'
    except IndexError:
        print("")
#Testing the model
recommend(2)
recommend(7)



# save the model to disk
FILENAME = 'finalized_model.sav'
pickle.dump(USERS_TFIDF, open(FILENAME, 'wb'))

# load the model from disk
LOADED_MODEL = pickle.load(open(FILENAME, 'rb'))

# Fetching the posts table
POSTS = pd.read_sql_query("select * from posts", ENGINE)

POSTS.head()

#Keeping only relevant columns
POSTS = POSTS.drop(['tags', 'slug', 'created_at', 'updated_at', 'image',
                    'status_id', 'action', 'post_id'], axis=1)

# filling the missing values with empty string and dropping empty short_bio rows
POSTS['title'] = POSTS['title'].fillna('')
POSTS.drop(POSTS[POSTS['title'] == ''].index, inplace=True)

#Taking out special characters
POSTS['title',] = POSTS['title'].str.replace(r'\W', ' ')
POSTS['content',] = POSTS['content'].str.replace(r'\W', ' ')
# Remove html tags
POSTS['content'] = POSTS['content'].str.replace(r'<[^>]*>', '')

# Remove white spaces including new lines
POSTS['content'] = POSTS['content'].str.replace(r'\s', ' ')

# Remove square brackets
POSTS['content'] = POSTS['content'].str.replace(r'\[.*?\]', '')

# Remove image files
POSTS['content'] = POSTS['content'].str.replace(r'\(.*?\)', '')

# Importing

POSTS_TFIDF = TfidfVectorizer(stop_words='english')

# computing TF-IDF matrix required for calculating cosine similarity
POSTS_TRANSF = POSTS_TFIDF.fit_transform(POSTS['title'])

COSINE_SIMILARITY = linear_kernel(POSTS_TRANSF, POSTS_TRANSF)

# running the function that would use our model to recommend articles to a user

recommend(4)
recommend(6)

#Loading our model


# save the model to disk
FILENAME = 'final_model.sav'
pickle.dump(POSTS_TFIDF, open(FILENAME, 'wb'))

# load the model from disk
LOADED_MODEL = pickle.load(open(FILENAME, 'rb'))
