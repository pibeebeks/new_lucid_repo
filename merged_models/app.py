''' Lucid blog recommender systems'''

# Import Relevant Modules
try:
    import nltk
    import re
    import pickle
    import joblib
    import requests
    from sklearn.metrics.pairwise import linear_kernel
    from flask import Flask, render_template, request, jsonify
    import pandas as pd
    from pandas.io.json import json_normalize
    from nltk.tokenize import sent_tokenize
    from nltk.corpus import stopwords
    from bs4 import BeautifulSoup
    from gensim.summarization import summarize
except ImportError as i_error:
    print(i_error)

# Load the datasets needed for deployment
USERS = pd.read_csv('used_data/users.csv')
USERS['name'] = USERS.name.str.lower()
USERS_SIM = pd.read_csv('used_data/users_sim.csv')
USERS_SIM['name'] = USERS_SIM.name.str.lower()
POSTS = pd.read_csv('used_data/posts_deploy.csv', index_col=0)
INDICES = pd.Series(USERS_SIM['user_id'].index)

# Download nltk dependencies
nltk.download('stopwords')
nltk.download('punkt')

# load the model from disk
MODEL = joblib.load('popular.sav')
SIMILAR_MODEL = pickle.load(open('finalized_model.sav', 'rb'))
ARTICLE_MODEL = pickle.load(open('final_model.sav', 'rb'))

# computing TF-IDF matrix required for calculating cosine similaritIES
USERS_TRANSF = SIMILAR_MODEL.fit_transform(USERS_SIM['short_bio'])
COSINE_SIMILARITY = linear_kernel(USERS_TRANSF, USERS_TRANSF)
POSTS_TRANSF = ARTICLE_MODEL.fit_transform(POSTS['title'])
COS_SIMILARITY = linear_kernel(POSTS_TRANSF, POSTS_TRANSF)

def recommend(index, cosine_sim=COSINE_SIMILARITY):
    '''Declaring a function that would use our model to fetch users
    similar to a given user based on user_bio'''
    try:
        i_d = INDICES[index]
        # Get the pairwsie similarity scores of all names
        # sorting them and getting top 10
        similarity_scores = list(enumerate(cosine_sim[i_d]))
        similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
        similarity_scores = similarity_scores[1:11]

        # Get the names index
        users_index = [i[0] for i in similarity_scores]

        # Return the top 10 most similar names
        return USERS_SIM['name'].iloc[users_index]
    except KeyError:
        return 'Invalid User ID, Enter a valid User Id'
    except IndexError:
        return 'This user has no bio'

def post_recommend(index, cosine_sim=COS_SIMILARITY):
    '''Function to recommend articles to users'''
    try:
        i_d = INDICES[index]
        # Get the pairwsie similarity scores of all names
        # sorting them and getting top 10
        similarity_scores = list(enumerate(cosine_sim[i_d]))
        similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
        similarity_scores = similarity_scores[1:11]

        # Get the names index
        posts_index = [i[0] for i in similarity_scores]

        # Return the top 10 most similar names
        return POSTS['content'].iloc[posts_index]
    except KeyError:
        return 'This user has no bio description/no article recommendation'
    except IndexError:
        print("")

# Initialize the app
app = Flask(__name__)

'''HTML GUI User Recommendation'''

# Render the home page
@app.route('/')
def home():
    '''Display of web app homepage'''
    return render_template('index.html')

# render the new_user_recommend page
@app.route('/new_user_recommend')
def new_user_recommend():
    '''Display of new user recommendation page'''
    return render_template('new_user_recommend_form.html')

# render the similar_user_recommend page
@app.route('/similar_user_recommend')
def similar_user_recommend():
    '''Display of similar user recommendation page'''
    return render_template('similar_user_recommend_form.html')

@app.route('/article_user_recommend')
def article_user_recommend():
    '''Display of similar user recommendation page'''
    return render_template('article_recommend_form.html')

# render the new user recommend page
@app.route('/recommend', methods=['POST'])
def new_user_recommender():
    '''Function that accepts the features and predicts
    a response(user recommendation) and displays it
    for Web App Testing'''
    # get the values from the form
    try:
        name_of_user = [x for x in request.form.values()]
        for name in name_of_user:
            name_of_user = name.lower()

        # recommend
        popular_users = MODEL.recommend((USERS[USERS.name == name_of_user]['name']).index[0])

        recommended_users = []
        for i_d in popular_users['user_id']:
            recommended_users.append(USERS.iloc[i_d, 0])
        return render_template('recommend.html', prediction_text=recommended_users)
    except:
        return render_template('recommend.html', prediction_text=["User does not exist"])

# render the recommended results
@app.route('/similar_recommend', methods=['POST'])
def similar_user_recommender():
    '''Function that accepts the features and predicts
    a response(user recommendation) and displays it
    for Web App Testing'''
    try:
        # get the values from the form
        name_of_user = [x for x in request.form.values()]
        for name in name_of_user:
            name_of_user = name.lower()
        # recommend
        similar_users = recommend(int(USERS_SIM[USERS_SIM.name == name_of_user]['user_id']))

        return render_template('similar_recommend.html', prediction_text=similar_users)
    except:
        text = ["User does not exist/Has no bio"]
        return render_template('similar_recommend.html', prediction_text=text)

# render the recommended results
@app.route('/post_recommend', methods=['POST'])
def article_user_recommender():
    '''Function that accepts the features and predicts
    a response(user recommendation) and displays it
    for Web App Testing'''
    try:
        # get the values from the form
        name_of_user = [x for x in request.form.values()]
        for name in name_of_user:
            name_of_user = name.lower()
        # recommend
        recommended_posts = post_recommend((USERS[USERS.name == name_of_user]['name']).index[0])
        return render_template('article_recommend.html', prediction_text=recommended_posts)
    except KeyError:
        return 'This user has no bio description/no article recommendation'
    except IndexError:
        return render_template('article_recommend.html', prediction_text=["User does not exist"])

@app.route("/similar_user_recommend_api", methods=['POST'])
def similar_user_recommend_api():
    '''Function that handles direct api calls
    from another client to recommend similar users'''
    try:
        json_data = request.get_json(force=True)
        json_df = json_normalize(json_data)
        for name in json_df.name:
            name_of_user = name.lower()
        recommended_users = recommend(int(USERS_SIM[USERS_SIM.name == name_of_user]['user_id']))
        
        final_recommendation = []

        for user in recommended_users:
            user_image = USERS.loc[USERS.name==user, 'image'].item()
            final_recommendation.append([user, user_image])

        recommended_users = {
            "recommended_users": [x for x in final_recommendation],
        }
        return jsonify(recommended_users)
    except:
        print("http error")

@app.route("/new_user_recommend_api", methods=['POST'])
def new_user_recommend_api():
    '''Function that handles direct api calls
    from another client to recommend Most Popular users'''
    try:
        json_data = request.get_json(force=True)
        json_df = json_normalize(json_data)
        for name in json_df.name:
            name_of_user = name.lower()
        popular_users = MODEL.recommend((USERS[USERS.name == name_of_user]['name']).index[0])

        recommended_users = []
        for i_d in popular_users['user_id']:
            recommended_users.append(USERS.iloc[i_d, 0])
        
        final_recommendation = []
        for user in recommended_users:
            user_image = USERS.loc[USERS.name==user, 'image'].item()
            final_recommendation.append([user, user_image])

        recommended_users = { 
            "recommended_users": [x for x in final_recommendation]
        }
        return jsonify(recommended_users)
    except:
        print("Server error")

@app.route("/article_recommend_api", methods=['POST'])
def article_user_recommend_api():
    '''Function that handles direct api calls
    from another client to recommend articles'''
    json_data = request.get_json(force=True)
    json_df = json_normalize(json_data)
    for name in json_df.name:
        name_of_user = name.lower()
    recommended_posts = post_recommend(int(USERS_SIM[USERS_SIM.name == name_of_user]['user_id']))
    recommended_posts = {
        "recommended_posts": [x for x in recommended_posts]
    }
    return jsonify(recommended_posts)

'''
page summarizer

'''
@app.route('/url_page_summarize')
def url_page_summarize():
    ''' Renders the page that
        will take in the url for summary
        in the web app'''
    return render_template('url_page_summarize.html')

@app.route('/article_summarize')
def article_summarize():
    ''' Renders the page that
        will take in the article for summary
        in the web app'''
    return render_template('article_summarize.html')

@app.route('/summarized_article', methods=['POST'])
def summarized_article():
    ''' Function that takes in an article
    and generates term frequencies and document frequencies
    and then ranks them as important terms which is used
    as a summary'''
    try:
        text_string = [x for x in request.form.values()]
        for texts in text_string:
            text = texts.lower()
        
        
        if len(text) >= 800:
            summary = summarize(text, ratio=0.1)
        else:
            summary = summarize(text, ratio=0.2)
        return render_template('summarized_url.html', summary=summary)
    except:
        return render_template('summarized_url.html', summary="Invalid Text")

@app.route('/summarized_url', methods=['POST'])
def summarized_url():
    ''' Function that takes in a url and reads the text
    and uses beautifulsoup to parse the html and extract
    text data from the p tags and generates term
    and document frequencies and then ranks them as
    important terms which is used as a summary
    '''
    try:
        text_array = []
        url_string = [x for x in request.form.values()]
        for urls in url_string:
            url = urls.lower()
        source = requests.get(url).text
        soup = BeautifulSoup(source, 'lxml')
        for paragraph in soup.find_all('p'):
            paragraph = paragraph.text
            text_array.append(paragraph)
        text = " ".join(text_array)
        
        if len(text) >= 800:
            summary = summarize(text, ratio=0.1)
        else:
            summary = summarize(text, ratio=0.2)
        
        return render_template('summarized_url.html', summary=summary)
    except:
        return render_template('summarized_url.html', summary="Invalid url/Url not found/server error")


@app.route('/summarized_article_api', methods=['POST'])
def summarized_article_api():
    ''' Function that takes in an article as a json object
    and generates term frequencies and document frequencies
    and then ranks them as important terms which is used
    as a summary.
    This function is  the api endpoint which will be called
    and it takes in data in the format {'text': 'the full text'}
    '''
    try:
        json_data = request.get_json(force=True)
        json_df = json_normalize(json_data)
        for texts in json_df.text:
            text = texts.lower()
        text = re.sub("[^a-zA-Z0-9.]", " ", text)

        if len(text) >= 800:
            summary = summarize(text, ratio=0.1)
        else:
            summary = summarize(text, ratio=0.2)
        return jsonify(summary)
    except:
        return jsonify("Invalid Text")


@app.route('/summarized_url_api', methods=['POST'])
def summarized_url_api():
    ''' Function that takes in a url as a json object
    and uses beautifulsoup to parse the html and extract
    text data from the p tags and generates term
    frequencies and document frequencies and then ranks
    them as important terms which is used as a summary.
    This function is  the api endpoint which will be called
    and it takes in data in the format {'url': 'the url '}
    '''
    try:
        text_array = []
        json_data = request.get_json(force=True)
        json_df = json_normalize(json_data)
        for urls in json_df.url:
            url = urls.lower()
        source = requests.get(url).text
        soup = BeautifulSoup(source, 'lxml')
        for paragraph in soup.find_all('p'):
            paragraph = paragraph.text
            text_array.append(paragraph)
        text = " ".join(text_array)

        if len(text) >= 800:
            summary = summarize(text, ratio=0.1)
        else:
            summary = summarize(text, ratio=0.2)
        return jsonify(summary)
    except:
        return jsonify("Invalid url/Url not found/server error")

# run the app
if __name__ == "__main__":
    app.run(debug=True)
