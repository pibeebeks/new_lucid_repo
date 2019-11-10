# new_lucid_repo

This a repo for the machine learning team of lucid.blog of the HNG Organization.
The repo contains machine learning tasks created to improve lucid.blog.
The tasks include:

1. Building a user recommender system for lucid.blog. The recommender systems
recommend users to people who share similar bio. The technique used for this
was the TFIDF vectorizer to vectorize the bio and get the similarites in the 
bios using cosine similarity matrix. The next user recommender system was designed
for new users or users without bios, the system recommends the most popular users
which was calculated based on interactions like followings, likes, loves and comments
on their posts.

2. Building a page summarizer for lucid.blog. The page summarizer can take a URL and summarize 
the page or take a text and sumarizes the text. The page summarizer doesn't summarize short text
because it doesn't have enought information to extract a summary. The technique used for building the
summarizer is the Extractive method of summarization. 

3. Building a system that monitors posts on the lucid blog and calculates its interest coefficient.
The grading method used to determine how interesting a post is, is the number of interactions the post had.

These systems were deployed to the cloud as both web apps and apis to be consumed by lucid.blog

The URL for the apps are 
```
https://lucid-ml.ml/ and https://lucidfeeds.herokuapp.com

```

### GUI web app testing of the recommender systems
For web app, Go to your browser and enter this url and test the recommender systems and summarizer, 
url for the web app: https://lucid-ml.ml/


For direct API calls, the following are the api urls
api url for popular users recommendation: https://lucid-ml.ml/new_user_recommend_api
api url for similar user's recommendation: https://lucid-ml.ml/similar_user_recommend_api
api url for Article recommendation: https://lucid-ml.ml/article_recommend_api
api url endpoint for url page summarizations: https://lucid-ml.ml/summarized_url_api
api url endpoint for article summarizations: https://lucid-ml.ml/summarized_article_api

For direct api calls, postman can be used for testing before production


### Documentation for the apis to be integrated to the lucid blog

https://documenter.getpostman.com/view/9109659/SVzxZLTK

https://documenter.getpostman.com/view/9109659/SVzxZLTL

https://documenter.getpostman.com/view/9109659/SVzxZLTM

### Interesting feeds testing 
To test the interesting feeds via a gui, the url of the web app is https://lucidfeeds.herokuapp.com

### interesting feeds documentation 
The interesting feeds api Documentation can be accessed from the web app of the interesting feeds at https://lucidfeeds.herokuapp.com/api_documentation

