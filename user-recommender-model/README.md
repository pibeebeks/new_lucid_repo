# User Recommender System 
<b>About:</b> 
This is a user recommender system, It was built using Python, focusing on the tasks 11,12,13,14,15,16,17,18,22 and 25


<b>List of libraries to be installed and how to install them</b>

scikit-learn - pip install scikit-learn<br/>
joblib - pip install joblib<br/>
pandas - pip install pandas<br/>
argparse - pip install argparse<br/>
numpy - pip install numpy<br/>
mysql.connector - pip install mysql.connector<br/>
sqlalchemy - pip install sqlalchemy<br/>
Requirements for the User recommender system hosted on heroku can be found in the requirements.txt in the Model_Deployment folder, click <a href="https://github.com/pibeebeks/task-user-recommender/blob/master/Model_Deployment/requirements.txt">here</a> <br />

<b>Running/Deployment:</b>

Task 14: Open Command Prompt in the folder and Call "the user_recommender.py  <user_id>" in the command line.
For example if we want to get the user recommendations for a particular user with user_id 7 we'd call the app as such
python user_recommended.py 7<br/>
Task 22: <br />
For GUI web app testing<br />
For web app, click https://lucidrecommendation.herokuapp.com/

For direct API calls, the following are the api urls<br />
api url for popular users recommendation: https://lucidrecommendation.herokuapp.com/new_user_recommend_api <br /> 
api url for similar user's recommendation: https://lucidrecommendation.herokuapp.com/similar_user_recommend_api <br /> 
api url for Article recommendation: https://lucidrecommendation.herokuapp.com/article_recommend_api 

For direct api calls, postman can be used for testing before production, postman can be downloaded <a href="https://www.getpostman.com/downloads/">here</a>

How to call the api, create a json object in this format {"name": "name of the user"} and call the api <br /> 

It can be tested by importing the following postman collections in the postman app <br />
https://www.getpostman.com/collections/46f5401336e7c29af100
https://www.getpostman.com/collections/06b621578adb09178ea0
https://www.getpostman.com/collections/b79eca9517ade75d3799

