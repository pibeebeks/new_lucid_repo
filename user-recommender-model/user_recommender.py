''' Importing utilities '''
try:
    import argparse
    import pandas as pd
    import mysql.connector
    from sqlalchemy import create_engine
    from sklearn.metrics.pairwise import linear_kernel
    from sklearn.feature_extraction.text import TfidfVectorizer

except ImportError as i_error:
    print(i_error)

# Getting the data from the database and showing the tables
try:
    MY_DB = mysql.connector.connect(host="remotemysql.com",
                                    user="8SawWhnha4",
                                    passwd="zFvOBIqbIz",
                                    database="8SawWhnha4")

except ConnectionError:
    print("Connection error")


    DBCURSOR = MY_DB.cursor()

    # Extracting the table relevant for users and their bio
    R = 'mysql+mysqlconnector://8SawWhnha4:zFvOBIqbIz@remotemysql.com/8SawWhnha4'
    ENGINE = create_engine(R)

    # Looking at the Users table
    USERS = pd.read_sql_query("select * from users", ENGINE)

    # Dropping columns considered irrelevant
    USERS = USERS.drop(["username", "email", "image", "provider", "provider_id", "password",
                        "remember_token", "created_at", "updated_at"], axis=1)

    # Checking other tables showed that id here means user_id
    # So we rename the id column to user_id

    USERS.rename(columns={"id": "user_id"}, inplace=True)
    # Construct a reverse map of indices and user name
    INDICIES = pd.Series(USERS.index, index=USERS['user_id'])
    USERS = USERS.set_index('user_id')

    # filling the missing values with empty string
    USERS['short_bio'] = USERS['short_bio'].fillna('')

    # removing Users without a bio
    USERS.drop(USERS[USERS['short_bio'] == ''].index, inplace=True)

    # Cleaning user_bio data
    USERS['short_bio'] = USERS['short_bio'].str.replace(r'\W', ' ')

    # Creating user TfidVectorizer object
    USERS_TFIDF = TfidfVectorizer(stop_words='english')

    # computing TF-IDF matrix required for calculating cosine similarity
    USERS_TRANSFORM = USERS_TFIDF.fit_transform(USERS['short_bio'])

    COSINE_SIMILARITY = linear_kernel(USERS_TRANSFORM, USERS_TRANSFORM)

def recommend(index, cosine_sim=COSINE_SIMILARITY):
    """
     Declaring a function that would use our model to fetch users
     similar to a given user based on user_bio
    :param index: User Id of user to recommend followers to
    :param cosine_sim: cosine similarity matrix
    :return: Pandas series of similar users based on bio
    """
    try:
        idx = INDICIES[index]
        # Get the pairwise similarity scores of all names
        # sorting them and getting top 10
        print(f"Getting recommendations for {USERS['name'].iloc[idx]}")
        similarity_scores = list(enumerate(cosine_sim[idx]))
        similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
        similarity_scores = similarity_scores[1:11]

        # Get the names index
        lucid_index = [i[0] for i in similarity_scores]

        #Return the top 10 most similar names

    except KeyError:
        return 'Invalid User ID, Enter a valid User Id'
    except IndexError:
        return 'This user has no bio'

    return USERS['name'].iloc[lucid_index]

def main():
    '''Function that returns the recommendations'''
    parser = argparse.ArgumentParser()
    parser.add_argument('user_id', type=int,
                        help='Enter User id of the user you wish to recommend followers to')

    args = parser.parse_args()

    recommendation_df = recommend(args.user_id)
    print(recommendation_df)


if __name__ == '__main__':
    main()
