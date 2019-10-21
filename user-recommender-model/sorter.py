" " " importation of pandas module" " "
try:
    import pandas as pd   #import the needed module
except ImportError as i_error:
    print(i_error)
def task12():         #define the function
    '''Function to define user preference.
    Returns output document with title of article and/or short bio of user
    '''
    try:
        d_f = pd.read_csv('posts.csv', usecols=['user_id', 'title', 'content'],
                          encoding="utf-8")#import the posts.csv to collect title
        d_h = d_f.sort_values(by='user_id')#sort in ascending order by user_id
        d_g = pd.read_csv('nooffollowers.csv', encoding="utf-8",
                          usecols=['user_id', 'username'])#reading nooffollowers file
        d_y = d_g.merge(d_h, how='left', on='user_id') #we merge our existing csvs
        dy1 = d_y[['user_id', 'username', 'title']]  #choose the relevant fields we need
        d_z = pd.read_csv('users.csv', usecols=['username', 'short_bio'],
                          encoding="utf-8") #we read the file which contains short bio
        d_p = d_z.merge(dy1, how='left', on='username') #we merge by username
        d_p.to_csv('output.csv', encoding='utf-8')#we define the name of the output
    except IOError as io_error:
        print(io_error)
task12()  #we call the function
