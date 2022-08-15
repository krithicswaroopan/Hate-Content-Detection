import snscrape.modules.twitter as sntwitter
import pandas as pd
import re
import unidecode

def getTweets(url, limit):
    pattern = "^https?:\/\/(?:www\.)?twitter\.com\/(?:#!\/)?@?([^/?#]*)(?:[?#].*)?$"
    username = re.findall(pattern, url)
    tweets = []
    un = 'from:' + username[0]
    for i,tweet in enumerate(sntwitter.TwitterSearchScraper(un).get_items()): #declare a username 
        if i>limit:
            break
        tweets.append([tweet.date, tweet.id, tweet.content, tweet.username]) #declare the attributes to be returned
    
    tweets_df = pd.DataFrame(tweets, columns=['Datetime', 'Tweet Id', 'Text', 'Username'])
    
    return username, tweets_df


def predictdf(model, df):
    df['Text'] = df['Text'].apply(lambda x : ' '.join([tweet for tweet in x.split()if not tweet.startswith("@")]))
    df['Text'] = df['Text'].apply(lambda x : ' '.join([tweet for tweet in x.split() if not tweet == '\d*']))
    df['Text'] = df['Text'].apply(lambda x : ' '.join([unidecode.unidecode(word) for word in x.split()])) 
    
    df['label'] = model.predict(df['Text'])
    
    return df


def predictText(model, Text):
    result = model.predict(Text) 
    return result
