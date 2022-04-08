#!/usr/bin/env python
# coding: utf-8

# In[4]:


# Importing all the relevant packages
import pandas as pd
import numpy as np
import os
import csv
from googletrans import Translator
import re
import nltk
from bs4 import BeautifulSoup
import unidecode
from nltk.stem import WordNetLemmatizer
import spacy
from spacy import displacy
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix, classification_report
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from nltk.corpus import sentiwordnet as swn
import text2emotion as te
from collections import Counter
from PIL import Image
from wordcloud import WordCloud
import dash
import plotly.express as px  
from dash import Dash, dcc, html, Input, Output, dash_table, State, callback
import dash_bootstrap_components as dbc
from skimage import io
import snscrape
import snscrape.modules.twitter as sntwitter
from datetime import datetime, timedelta


# In[5]:


nltk.download('wordnet')


# In[6]:


nltk.download('punkt')


# In[11]:


from datetime import date
today = date.today()
today1=today.strftime('%Y-%m-%d')
last_week= today - timedelta(days=30)
last_week=last_week.strftime('%Y-%m-%d')
today=today.strftime("%m/%d/%Y")
today=today.replace("/","_")


# ### Social Media Scrapper

# In[12]:


# loading the file that contains the keywords for which the tweets will be extracted
df_terms = pd.read_csv ('tweet_terms.csv')


# In[13]:


# loop to take out all the tweets
j=0
df_finalscrap=pd.DataFrame()
while j<len(df_terms.Terms):
    tweets_list= []
    term= df_terms.Terms[j]
    # Using TwitterSearchScraper to scrape data and append tweets to list
    for i,tweet in enumerate(sntwitter.TwitterSearchScraper(term +' since:'+last_week+' until:'+today1+' near: India').get_items()):
        if i>100:
            break
        tweets_list.append([tweet.date, tweet.id, tweet.content, tweet.user.username, tweet.user.location, tweet.lang, tweet.source])
    # Creating a dataframe from the tweets list above
    tweets_df = pd.DataFrame(tweets_list, columns=['Datetime', 'Tweet Id', 'Text', 'Username', 'Location', 'Language', 'Source'])
    tweets_df['Keyword']=term
    df_finalscrap= pd.concat([df_finalscrap, tweets_df])
    j=j+1


# In[15]:


df_tweet_clean_terms = pd.read_csv ('tweet_clean_terms.csv')


# In[16]:


# remove junk tweets
i=0
while i<len(df_finalscrap.Text):
    j=0
    while j<len(df_tweet_clean_terms.Delete_term):
        delete_term= df_tweet_clean_terms.Delete_term[j]
        j+=1
        if (df_finalscrap.iloc[i,2]).find(delete_term)>-1:
            df_finalscrap=df_finalscrap.drop(df_finalscrap.index[i])
            i-=1
            break
    i+=1


# In[17]:


df_finalscrap.to_csv("Scrapped_tweets_" + today +".csv")


# ### Language Translation

# In[4]:


translator = Translator()


# In[19]:


df_tweets_raw = pd.read_csv ("Scrapped_tweets_" + today +".csv")


# In[6]:


# translating all tweets to english
i=0
l=len(df_tweets_raw.Text)
translated_tweets = []
while i<l:
    try:
        temp= translator.translate(df_tweets_raw.Text[i], dest='en')
        translated_tweets.append(temp.text)
    except:
        translated_tweets.append(df_tweets_raw.Text[i])
    # to check if it is iterating or not
    #print(i)  
    i=i+1


# In[7]:


df_tweets_raw['Translated']=translated_tweets


# In[9]:


# removing useless columns
df_tweets_raw.drop('Unnamed: 0', axis='columns', inplace=True)
df_tweets_raw.drop('Tweet Id', axis='columns', inplace=True)
df_tweets_raw.drop('Source', axis='columns', inplace=True)


# In[117]:


# Export dataframe into a CSV
df_tweets_raw.to_csv('translated_tweets_' + today +'.csv', sep=',', index=False)


# ### Extraction

# In[11]:


df_tweets_translated = df_tweets_raw


# In[12]:


# lower case function
def lowertext(text):
    x=text.lower()
    return x


# In[13]:


# Function to convert list to string
def listToString(s):  
    str1 = "" 
    for ele in s: 
        str1 += ele + " "  
    return str1 


# In[14]:


## email id extraction
def email_ext(text):
    email = re.findall('\S+@\S+', text)
    return email


# In[15]:


## email id removal
def email_rem(text):
    email = re.sub('\S+@\S+','', text)
    return email


# In[16]:


## phone no extraction
def phoneno_ext(text):
    phoneno= re.findall('((?:\+\d{2}[-\.\s]??|\d{4}[-\.\s]??)?(?:\d{3}[-\.\s]??\d{3}[-\.\s]??\d{4}|\(\d{3}\)\s*\d{3}[-\.\s]??\d{4}|\d{3}[-\.\s]??\d{4}))', text)
    return phoneno


# In[17]:


## phone no removal
def phoneno_rem(text):
    phoneno= re.sub('((?:\+\d{2}[-\.\s]??|\d{4}[-\.\s]??)?(?:\d{3}[-\.\s]??\d{3}[-\.\s]??\d{4}|\(\d{3}\)\s*\d{3}[-\.\s]??\d{4}|\d{3}[-\.\s]??\d{4}))','' ,text)
    return phoneno


# In[18]:


# removal of https, urls and other links
def urlbuster(thestring):
    URLless_string = re.sub(r'\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*', '', thestring)
    return(URLless_string)


# In[19]:


#extraction of hashtags
def extract_hashtags(text):
    hashtag_list = []
    for word in text.split():
        if word[0] == '#':
            hashtag_list.append(word[1:])
    return(hashtag_list)


# In[20]:


#removal of hashtags
def remove_hashtags(text):
    hashtag_list = []
    for word in text.split():
        if word[0] != '#':
            hashtag_list.append(word[0:])
    return(listToString(hashtag_list))


# In[21]:


#extraction of @ mentions
def extract_mention(text):
    mention_list = []
    for word in text.split():
        if word[0] == '@':
            mention_list.append(word[1:])
    return(mention_list)


# In[22]:


#removal of @ mentions
def remove_mention(text):
    mention_list = []
    for word in text.split():
        if word[0] != '@':
            mention_list.append(word[0:])
    return(listToString(mention_list))


# In[23]:


# normalization of accented characters 
def accento_correcto(text):
    output_string = unidecode.unidecode(text)
    return(output_string)


# In[24]:


# remove special characters, punctuations
def not_so_special_anymore(text):
    x=re.sub('[^A-Za-z0-9 ]+', ' ', text)
    return(x)


# In[25]:


# function to remove recurring spaces
def singularity(text):
    x=re.sub(' +', ' ', text)
    return(x)


# In[26]:


#df_tweets_translated.head(2)


# In[27]:


# Calling functions for extraction of email id, phone no, hashtags, mentions and removal of everything except stopwords
# Entity dealt seperately
i=0
l=len(df_tweets_translated.Translated)
cleaned_data = []
email_list=[]
phonebook=[]
hashtag_list =[]
mention_list=[]
while i<l:
    try:
        email_list.append(email_ext(df_tweets_translated.Translated[i]))
    except:
        email_list.append("")
    try:
        phonebook.append(phoneno_ext(df_tweets_translated.Translated[i]))
    except:
        phonebook.append("")
    try:
        mention_list.append(extract_mention(df_tweets_translated.Translated[i]))
    except:
        mention_list.append("")
    try:    
        hashtag_list.append(extract_hashtags(df_tweets_translated.Translated[i]))
    except:
        hashtag_list.append("")
     
    temp=(accento_correcto(df_tweets_translated.Translated[i]))
    temp=(remove_mention(temp))
    temp=(remove_hashtags(temp))
    temp=(phoneno_rem(temp))
    temp=(email_rem(temp))
    temp=(urlbuster(temp))
    temp=(singularity(temp))
    temp=(not_so_special_anymore(temp))
    cleaned_data.append(temp)          
    i=i+1


# ### Entity Detection

# In[28]:


NER = spacy.load("en_core_web_sm")


# In[29]:


# function to take out the entities
def inspector_entity_detector(string):
    text1=NER(string)
    temp=[]
    for word in text1.ents:
        if word.label_ == "GPE" or word.label_== "PERSON" or word.label_== "ORG":
            temp.append(word.text)
    return temp


# In[30]:


# calling the function to extract the entities
i=0
l=len(cleaned_data)
entity_set = []
while i<l:
    temp=[]
    try:
        temp= inspector_entity_detector(cleaned_data[i])
    except:
        temp=None
    entity_set.append(temp)
    #print(i)
    i=i+1


# In[32]:


df_tweets_translated['Entities']=entity_set


# In[34]:


# converting entity superset list of list to list for removal from tweets
entity_superlist=[]
l=len(entity_set)
i=0
while i<l:
    k=len(entity_set[i])
    j=0
    while j<k:
        temp=entity_set[i][j].lower()
        entity_superlist.append(temp)
        j+=1
    i+=1


# In[35]:


# converting entity superset list of list to list for word cloud later
entity_superlist_cloud=[]
l=len(entity_set)
i=0
while i<l:
    k=len(entity_set[i])
    j=0
    while j<k:
        temp=entity_set[i][j].lower()
        temp=re.sub(' +', ' ', temp)
        temp=(temp.strip())
        entity_superlist_cloud.append(temp)
        j+=1
    i+=1


# In[36]:


# convert list of entitites to dictionary with frequency without repetition
freq = {} 
for item in entity_superlist_cloud: 
    if (item in freq):         freq[item] += 1
    else: 
        freq[item] = 1


# In[38]:


# function removal of entities from tweets
def entity_buster(text):
    text_ls= text.split() 
    final_list = [x for x in text_ls if x not in entity_superlist]
    final_string= ' '.join(final_list)
    return final_string


# In[39]:


# removal of entities and case conversion
i=0
l=len(cleaned_data)
final_data = []
while i<l:
    
    try:
        temp= entity_buster(cleaned_data[i])
        final_data.append(lowertext(temp))
    except:
        final_data.append(lowertext(cleaned_data[i]))
    i=i+1


# In[40]:


#df_tweets_translated.head(2)


# In[41]:


df_tweets_translated['Email']=email_list
df_tweets_translated['Phone_no']=phonebook
df_tweets_translated['Hashtag']=hashtag_list
df_tweets_translated['Mentions']=mention_list
df_tweets_translated['data_w_stopwords']=final_data


# In[42]:


del df_tweets_translated['Language']
del df_tweets_translated['Translated']


# Splitting the data into 2 based on stop word removal

# In[43]:


# stopword removal
sw_spacy = NER.Defaults.stop_words
def stopword_remover(text):
    words = [word for word in text.split() if word.lower() not in sw_spacy]
    new_text = " ".join(words)
    return new_text


# In[44]:


# spacy word tokenizer
def token_of_appreciation(texts):
    doc=NER(texts)
    tokens = []
    for token in doc:
        tokens.append(token.text)
    return(tokens)


# In[45]:


# spacy lemmatization
lemmatizer = WordNetLemmatizer()
def lemon_lemmatizer(texts):
    # Tokenize: Split the sentence into words
    word_list = nltk.word_tokenize(texts)
    # Lemmatize list of words and join
    lemmatized_output = ' '.join([lemmatizer.lemmatize(w) for w in word_list])
    return(lemmatized_output)


# In[47]:


# removal of stopwords
i=0
l=len(df_tweets_translated.data_w_stopwords)
wo_stopwords = []
while i<l:
    try:
        wo_stopwords.append(stopword_remover(df_tweets_translated.data_w_stopwords[i]))
    except:
        wo_stopwords.append(df_tweets_translated.data_w_stopwords[i])
    i=i+1


# In[48]:


df_tweets_translated['data_wo_stopwords']=wo_stopwords


# ### Sentiment Analysis and Emotion Classification

# In[50]:


# POS tagger dictionary
pos_dict = {'J':wordnet.ADJ, 'V':wordnet.VERB, 'N':wordnet.NOUN, 'R':wordnet.ADV}

def token_stop_pos(text):
    tags = pos_tag(word_tokenize(text))
    newlist = []
    for word, tag in tags:
        if word.lower() not in set(stopwords.words('english')):
            newlist.append(tuple([word, pos_dict.get(tag[0])]))
    return newlist

df_tweets_translated['POS tagged'] = df_tweets_translated['data_w_stopwords'].apply(token_stop_pos)
#df_tweets_translated.head()


# In[51]:


# lemmatizing words to their lemmas
wordnet_lemmatizer = WordNetLemmatizer()
def lemmatize(pos_data):
    lemma_rew = " "
    for word, pos in pos_data:
        if not pos:
            lemma = word
            lemma_rew = lemma_rew + " " + lemma
        else:
            lemma = wordnet_lemmatizer.lemmatize(word, pos=pos)
            lemma_rew = lemma_rew + " " + lemma
    return lemma_rew

df_tweets_translated['Lemma'] = df_tweets_translated['POS tagged'].apply(lemmatize)
#df_tweets_translated.head()


# In[52]:


fin_data = pd.DataFrame(df_tweets_translated[['Text', 'Lemma']])


# In[53]:


# assigning the sentiments
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyzer = SentimentIntensityAnalyzer()
# function to calculate vader sentiment
def darthvadersentimentanalysis(review):
    vs = analyzer.polarity_scores(review)
    return vs['compound']

fin_data['Vader Sentiment'] = fin_data['Lemma'].apply(darthvadersentimentanalysis)
# function to analyse
def darthvader_analysis(compound):
    if compound >= 0.5:
        return 'Positive'
    elif compound <= -0.5 :
        return 'Negative'
    else:
        return 'Neutral'
fin_data['Vader Analysis'] = fin_data['Vader Sentiment'].apply(darthvader_analysis)
#fin_data.head()


# In[54]:


vader_counts = fin_data['Vader Analysis'].value_counts()
#vader_counts


# In[55]:


# Emotional analysis
# removal of stopwords
i=0
l=len(df_tweets_translated.data_w_stopwords)
emotions = []
while i<l:
    q=te.get_emotion(df_tweets_translated.data_w_stopwords[i])
    k = Counter(q)
    high = k.most_common(2)
    emotions.append(high)
    i=i+1


# In[56]:


fin_data['Emotions']=emotions


# In[57]:


df_tweets_translated['Sentiment']=fin_data['Vader Analysis']


# In[58]:


df_tweets_translated['Emotion']=emotions


# In[59]:


del df_tweets_translated['Lemma']
del df_tweets_translated['POS tagged']


# In[118]:


# Export dataframe into a CSV
df_tweets_translated.to_csv('Analysis_file_' + today +'.csv', sep=',', index=False)


# ### Things to be included in Interactive Dashboard
# 1. EDA (Exploratory data analysis)
#     1.1 Total no of tweets analysed in the run
#     1.2 Geographic distribution
# 2. Type of fraud/ payment system in question
# 3. Overall sentiment (Vader graph)
# 4. Entity word cloud (after removing geographical locations)
# 5. Emotion superset pie chart
# 6. Button see historical entity data and see changes
# 7. Button to see the complete final analysed data csv

# In[119]:


df_graphing = pd.read_csv ('Analysis_file_' + today +'.csv')
df_graphing.head(1)


# In[97]:


del df_graphing['data_w_stopwords']
del df_graphing['data_wo_stopwords']
del df_graphing['Username']
del df_graphing['Hashtag']


# In[63]:


# assigning severity scores to tweets based on detected sentiment and emotions
i=0
score_list=[]
classification_list=[]
while i<len(df_graphing.Emotion):
    score123=0
    if df_graphing.Sentiment[i]=="Negative":
        score123+=1
    elif df_graphing.Sentiment[i]=="Positive":
        score123-=1
    else:
        score123+=0

    if df_graphing.Emotion[i].find("Fear")>-1:
        score123+=2
    if df_graphing.Emotion[i].find("Sad")>-1:
        score123+=1
    if df_graphing.Emotion[i].find("Angry")>-1:
        score123+=2
    if df_graphing.Emotion[i].find("Happy")>-1:
        score123-=1
    if df_graphing.Emotion[i].find("Suprise")>-1:
        score123+=0
    score_list.append(score123)
    if score123>2:
        classification_list.append("Complaint")
    elif score123>-1:
        classification_list.append("Suggestions & Recommendations")
    else:
        classification_list.append("Positive Experiences")
    i+=1


# In[64]:


df_graphing['Severity']=score_list
df_graphing['Classfication']=classification_list


# In[65]:


#EDA
#totwal no of tweets
total_tweet_count= len(df_graphing.Text)

# geographical distribution
tweets_geo=df_graphing.Location
cleanedList = [x for x in tweets_geo if str(x) != 'nan']

# convert list of location to dictionary with frequency without repetition
freq_geo = {} 
for item in cleanedList: 
    if (item in freq_geo):         freq_geo[item] += 1
    else: 
        freq_geo[item] = 1

counter_geo = Counter(freq_geo)
high_geo= counter_geo.most_common(10)
high_geo_dict = dict(high_geo)


# In[66]:


# bar chart for top 10 locations of tweets
names_geo = list(high_geo_dict.keys())
values_geo = list(high_geo_dict.values())


# In[67]:


# convert list of type of fraud to dictionary with frequency without repetition
freq_keyword = {} 
for item in df_graphing.Keyword: 
    if (item in freq_keyword):         freq_keyword[item] += 1
    else: 
        freq_keyword[item] = 1


# In[68]:


# Data to plot
labels_type = []
sizes_type = []

for x, y in freq_keyword.items():
    labels_type.append(x)
    sizes_type.append(y)


# In[69]:


# convert list of Classification to dictionary with frequency without repetition
freq_keyword_class = {} 
for item in df_graphing.Classfication: 
    if (item in freq_keyword_class):         freq_keyword_class[item] += 1
    else: 
        freq_keyword_class[item] = 1


# In[70]:


# Data to plot
labels_type_class = []
sizes_type_class = []

for x, y in freq_keyword_class.items():
    labels_type_class.append(x)
    sizes_type_class.append(y)


# In[71]:


# making a copy of
freq_cloud=freq.copy()


# In[72]:


# reads the entity cleaner file and converts into a list
df_entity_cleaner = pd.read_csv ('entity_cleaner.csv')
entity_cleaner_list=df_entity_cleaner.Remove.tolist()


# In[73]:


# cleaning entities based on list
dels = []
for k, v in freq_cloud.items():
    if k in entity_cleaner_list:
        dels.append(k)

for i in dels:
    del freq_cloud[i]


# In[74]:


# creating of world cloud and saving it as image for dashboard
wordcloud = WordCloud(width = 500, height = 500).generate_from_frequencies(freq_cloud)

wordcloud =  wordcloud.to_file('Entity_cloud.png')


# In[75]:


# Emotions pie chart
emotion_list_graph=[]
emotion_list_graph=df_graphing.Emotion.copy()
emotion_string_graph= emotion_list_graph.to_string()
emotion_string_graph=re.sub(r'[^a-zA-Z]', ' ', emotion_string_graph)
emotion_string_graph=re.sub(' +', ' ', emotion_string_graph)
emotions_tokens = nltk.word_tokenize(emotion_string_graph)


# In[76]:


# convert list of entitites to dictionary with frequency without repetition
freq_emotion= {} 
for item in emotions_tokens: 
    if (item in freq_emotion):         freq_emotion[item] += 1
    else: 
        freq_emotion[item] = 1


# In[78]:


# Data to plot
labels_emo = []
sizes_emo = []

for x, y in freq_emotion.items():
    labels_emo.append(x)
    sizes_emo.append(y)


# ### Dashboard

# In[109]:


app = Dash(__name__)
# App layout


url_bar_and_content_div = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content')
])

layout_index = html.Div([
    dcc.Textarea(
        id='textarea-example',
        value='Select one of the options from below to navigate to either see the data with analysis columns or see the analytical graphs for overview ',
        style={'width': '80%', 'height': 50},
    ),
    html.Div(id='textarea-example-output', style={'whiteSpace': 'pre-line'}),
    html.Br(),
    dcc.Link('Navigate to Data page', href='/page-1'),
    html.Br(),
    dcc.Link('Navigate to Analysis', href='/page-2'),
])

layout_page_1 = html.Div([
    html.H2('Data View'),
    html.Br(),
    dcc.Link('Navigate to "/"', href='/'),
    html.Br(),
    dcc.Link('Navigate to Analysis', href='/page-2'),
    html.Br(),
    dcc.Textarea(
        id='textarea-example',
        value='Scroll Right to see all the columns. Scroll to the bottom to go to the next page (50 records per page). You can also sort and filter data from the top.',
        style={'width': '80%', 'height': 50},
    ),
    html.Div(id='textarea-example-output', style={'whiteSpace': 'pre-line'}),
    html.Br(),
    dash_table.DataTable(
        id='datatable-interactivity',
        columns=[
            {"name": i, "id": i, "deletable": False, "selectable": True} for i in df_graphing.columns
        ],
        data=df_graphing.to_dict('records'),
        editable=False,
        filter_action="native",
        sort_action="native",
        sort_mode="multi",
        row_deletable=False,
        page_action="native",
        page_current= 0,
        page_size= 50,
    ),
    html.Div(id='datatable-interactivity-container')
])

layout_page_2 = html.Div([
    html.H1("Fraud Detection using Social Media", style={'text-align': 'center'}),
    dcc.Dropdown(id="slct_year",
                 options=[
                     {"label": "Exploratory Data Analysis", "value": "Exploratory Data Analysis"},
                     {"label": "Type of Fraud and Payment Instrument", "value": "Type of Fraud and Payment Instrument"},
                     {"label": "Sentiment Analysis", "value": "Sentiment Analysis"},
                     {"label": "Emotion Classification", "value": "Emotion Classification"},
                     {"label": "Entity Detection", "value": "Entity Detection"},
                     {"label": "Classification of Tweets based on Intent", "value": "Intent"}],                     
                 multi=False,
                 value="Intent",
                 style={'width': "60%"},
                 placeholder="Select an analysis"
                 ),
    html.Div(id='output_container', children=[]),
    html.Br(),
    html.Div(id='page-2-display-value'),
    html.Br(),
    dcc.Link('Navigate to "/"', href='/'),
    html.Br(),
    dcc.Link('Navigate to Data page', href='/page-1'),
    html.Br(),
    dcc.Graph(id='my_bee_map', figure={})
])

# index layout
app.layout = url_bar_and_content_div

# "complete" layout
app.validation_layout = html.Div([
    url_bar_and_content_div,
    layout_index,
    layout_page_1,
    layout_page_2,
])


# Index callbacks
@app.callback(Output('page-content', 'children'),
              Input('url', 'pathname'))
def display_page(pathname):
    if pathname == "/page-1":
        return layout_page_1
    elif pathname == "/page-2":
        return layout_page_2
    else:
        return layout_index


# Page 1 callbacks
@app.callback(
    Output('datatable-interactivity', 'style_data_conditional'),
    Input('datatable-interactivity', 'selected_columns')
)
def update_styles(selected_columns):
    return [{
        'if': { 'column_id': i },
        'background_color': '#D2F3FF'
    } for i in selected_columns]


# Page 2 callbacks
@app.callback(
    [Output(component_id='output_container', component_property='children'),
     Output(component_id='my_bee_map', component_property='figure')],
    [Input(component_id='slct_year', component_property='value')]
)
def update_graph(option_slctd):
    print(option_slctd)
    print(type(option_slctd))
    
    container = "The analysis chosen is: {}".format(option_slctd)
    
    #fig = px.pie(df_graphing, values=vader_counts.values, names= vader_counts.index)

    
    if option_slctd == 'Emotion Classification':
        fig = px.pie(df_graphing, values=sizes_emo, names=labels_emo)    

    elif option_slctd == 'Exploratory Data Analysis':
        fig = px.pie(df_graphing, values=values_geo, names=names_geo)    
    
    elif option_slctd == 'Type of Fraud and Payment Instrument':
        fig = px.pie(df_graphing, values=sizes_type, names=labels_type)
        
    elif option_slctd == 'Intent':
        fig = px.pie(df_graphing, values=sizes_type_class, names=labels_type_class)
       
    elif option_slctd == 'Sentiment Analysis':
        fig = px.pie(df_graphing, values=vader_counts.values, names= vader_counts.index)
  
    elif option_slctd == 'Entity Detection':
        img = io.imread('Entity_cloud.png')
        fig = px.imshow(img)
        
    return container, fig


# In[110]:


if __name__ == '__main__':
    app.run_server(debug=False)


# In[ ]:




