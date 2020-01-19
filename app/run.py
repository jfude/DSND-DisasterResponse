import json
import plotly
import numpy as np
import pandas as pd
import re
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sqlalchemy import create_engine
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger','stopwords'])

app = Flask(__name__)

"""
def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens
"""


## Tokenizer needs to be absolutely identical to that found in classifier.pkl??
def tokenize(text):
    
    """
    Tokenize (Clean/Normalize etc...) text input

    Keyword Arguments:
    text: Message string to be tokenized

    Returns:
    clean_tokens: List of clean tokens

    """

    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    
    ## Combine these two lines as we did before
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")

    ## Remove punctuation  
    text = re.sub(r"[^a-zA-Z0-9]", " ",text)
    
    tokens = word_tokenize(text)
    ## Remove stop words
    tokens = [w for w in tokens if w not in stopwords.words("english")]
    
    lemmatizer = WordNetLemmatizer()

    ## Lemmatize on nouns ('n'), verbs ('v'), adjectives('a'), adverbs ('r')
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok,pos = 'n').lower().strip()
        clean_tok = lemmatizer.lemmatize(clean_tok,pos = 'v')
        clean_tok = lemmatizer.lemmatize(clean_tok,pos = 'a')
        clean_tok = lemmatizer.lemmatize(clean_tok,pos = 'r')
        clean_tokens.append(clean_tok)

    return clean_tokens




# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('DisasterResponse', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    ## Data for second visual
    ## Histogram of categories represented in the DisasterDatabase.db
    s = np.mean(df.iloc[:, 4:]).sort_values(ascending = False)
    category_percentage     = list(s) ## percent of category in table
    category_names          = list(s.keys())
               
    
    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },

        {
         
            'data': [
                Bar(
                    x=category_names,
                    y=category_percentage
                )
            ],

            'layout': {
                'title': 'Percentage of Message Categories in Database Represented',
                'yaxis': {
                    'title': "Percentage",
                    'tickformat':".1%"
                },
                'xaxis': {
                    'title': "Category",
                    'automargin': True
                },
                
            }


        }
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    
    # use model to predict classification for query
    classification_labels  = model.predict([query])[0]    
    classification_results = dict(zip(df.columns[4:], classification_labels))

    classification_probs   = model.predict_proba([query])
    
    classification_probs_results = {}
    
    
    for i in range(0,len(classification_labels)):
        if (classification_labels[i]==1.):
            classification_probs_results.update({df.columns[4+i]:classification_probs[i][0][1]})
    
    
    
    ## Sort dictionary 'classification_probs_results' by values
    classification_probs_results = {k: v for k, v in sorted(classification_probs_results.items(), key=lambda item: item[1])}
    
    classification_names = list(classification_probs_results.keys())
    classification_probs = list(classification_probs_results.values())
   
    
    ## Build plot of probabilities for categories that were found to be significant
    ## We plot the probability of getting 1.0 for the category (as opposed to 0.0)
    ## for significant categories. These are conventionally greater than about 60%
    graphs = [
        {
            'data': [
                Bar(
                    x=classification_probs,
                    y=classification_names,
                    orientation='h'
                )
            ],

            'layout': {
                'title': 'Classification Probabilities',
                'yaxis': {
                    'title': "Category"
                },
                'xaxis': {
                    'title': "Probability",
                    'tickformat':".1%",
                    'automargin': True
                }
                
            }
        }
    ]


    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    
    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results,
        ids = ids,
        graphJSON = graphJSON
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()
