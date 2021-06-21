import re

from nltk.stem import PorterStemmer
from wordcloud import STOPWORDS
from textblob import TextBlob
import pandas as pd
import numpy as np
import nltk

stopwords = set(STOPWORDS)


# returns sentiment score
def senti(x):
    return TextBlob(x).sentiment  

# precesses text column to get a sentiment score
def sentiment_analysis(df, col):
    # Change the col type to string
    df[col] = df[col].astype(str)
    
    # Lowercase all reviews
    df[col] = df[col].apply(lambda x: " ".join(x.lower() for x in x.split()))
    
    # Remove punctuation
    df[col] = df[col].str.replace('[^\w\s]','')
    
    # Remove stop words
    df[col] = df[col].apply(lambda x: " ".join(x for x in x.split() if x not in stopwords))
    
    # Sentiment score
    df[col+'_score'] = df[col].apply(senti)
  
    return df


# replace missing value in text column with a dummy variable & mask value 
def parse_missing_col_values(df, col):
    col_name = col + '_is_missing'
    
    df[col_name] = 0                                 # create a new column
    df.loc[df[col].isnull(), col_name] = 1           # set values to 1 where appropriate 
    df.loc[df[col].isnull(), col] = 'no ' + col      # replace Nan value with mask value
    
    #test
    val = df[df[col].isnull() == True].shape[0]
    if val == 0:
        print('Success!')
    else:
        print('Oops, something went wrong')
        
    return df


# label each bar directly
def label_each_bar(axes):
    for ax in axes:
        for p in ax.patches:
            ax.annotate(
                format(p.get_height(), '.2f'), 
                (p.get_x() + p.get_width() / 2., p.get_height()), 
                ha = 'center', 
                va = 'center', 
                xytext = (0, 10), 
                textcoords = 'offset points')
            
    pass


# remove spine on the plot
def get_rid_of_spine(axes):
    for ax in axes:
        for spine in ax.spines.values():
            spine.set_visible(False)
            
    pass
            
    
# update 't' and 'f' strings to 1 & 0 correspondingly
def update_tf_to_bool(df, cols):
    for col in cols:
        df[col] = pd.Series(np.where(df[col].values == 't', 1, 0), df.index)
        
    return df


# get most frequent words
def get_top_words(df, col_name, n=30):
    stopwords.update(['br', 'b', '<br>', '</br>'])
    
    words_dict = {}
    
    listings = df[col_name].tolist()
    
    for entry in listings:
        words = entry.lower().replace('<b>', '').replace('<br/>', '').replace('<br>', '').replace(' br ', '').replace(' bdr ', '')
        words = re.sub(r'[^a-zA-Z\s]+', '', words)
        words = words.split()

        for word in words:
            word = word.lower()

            # ignore stop words
            if word  in stopwords:
                continue;

            if word in words_dict:
                words_dict[word] += 1
            else:
                words_dict[word] = 1
                
    top_words = sorted(words_dict.items(), key=lambda item: item[1], reverse=True)[:n]
    return top_words


# remove matching
def get_diff(top_list, lower_list):
    
    all_top = remove_count(top_list)
    all_lower = remove_count(lower_list)
    
    unique_top = []
    unique_lower = []
    
    for word in all_top:
        if word in all_lower:
            continue
        else:
            unique_top.append(word)
            
    for word in all_lower:
        if word in all_top:
            continue
        else:
            unique_lower.append(word)
    
    
    return unique_top, unique_lower


def remove_count(l):
    words = []
    
    for pair in l:
        words.append(pair[0])
    
    return set(words)


def create_dummy_df(df, cat_cols, dummy_na):
    '''
    INPUT:
    df - pandas dataframe with categorical variables you want to dummy
    cat_cols - list of strings that are associated with names of the categorical columns
    dummy_na - Bool holding whether you want to dummy NA vals of categorical columns or not
    
    OUTPUT:
    df - a new dataframe that has the following characteristics:
            1. contains all columns that were not specified as categorical
            2. removes all the original columns in cat_cols
            3. dummy columns for each of the categorical columns in cat_cols
            4. if dummy_na is True - it also contains dummy columns for the NaN values
            5. Use a prefix of the column name with an underscore (_) for separating 
    '''
    for col in cat_cols:
        try:
            # for each cat add dummy var, drop original column
            df = pd.concat([
                df.drop(col, axis=1),
                pd.get_dummies(
                    df[col], drop_first=True, prefix=col, dummy_na=dummy_na)
            ], axis=1)
        except:
            continue
            
    return df

