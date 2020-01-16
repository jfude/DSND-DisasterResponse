import sys

import re
import pandas as pd
import numpy as np
import pickle

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.metrics import classification_report,accuracy_score 
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger','stopwords'])


def load_data(database_filepath):
    
    """
    Loads data from disaster response sqlite database. 
    Assumes name of the single table in the db is 
    DisasterResponse
    
    Keyword Arguments:
    Relative path to the database

    Returns:
    X: ndarray of message
    y: ndarray of ndarrays ('matrix'),  36 columns for each message category (label),
       each row corresponds to message from X. Cell of matrix is 1 if message corresponds
       to category indicated by column, 0 otherwise.

    categories: Name of each column in y

    """

    df =  pd.read_sql_table("DisasterResponse",'sqlite:///'+database_filepath)
    X = df.message.values ## gives a numpy array
    y = df.iloc[:, 4:].values ## gives a numpy array of numpy arrays
    categories = list(df.iloc[:,4:].columns)
    
    return X, y, categories






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




def build_model():
    
    
    """ 
    Construct model : Feature Extraction: TFidf for 
                      word collection/message category correspondence
                      Classifier: RandomForest
                      Use Grid Search to optimize hyperparameters of classifier

    
    Keyword Arguments: None

    Returns:
    model: Trained classifier model

    """


    pipeline = Pipeline([
                        ('vect', CountVectorizer(tokenizer=tokenize)),
                        ('tfidf', TfidfTransformer()),
                        ('clf' , MultiOutputClassifier(RandomForestClassifier()))
                        ])

    parameters = {'clf__estimator__n_estimators':[10,50,75],
                  'clf__estimator__min_samples_split':[4,8,12]
                 }
        
    model = GridSearchCV(pipeline, param_grid = parameters,cv=3)  

    return model
    

def evaluate_model(model, X_test, Y_test, category_names):
    
    """
    Evaluate model accuracy, print results
    
    Keyword Arguments:        
                                                                                
    model:  The input model which has a predict method 
    X_test: Test features (messages)
    Y_test: Test labels   (categories)
    category_names: names in labels in Y_test

    Returns: None                                                                                                                    

    """
    
    y_pred = model.predict(X_test)

    n_categories = len(category_names)
    for i in range(n_categories):
            print("\n\n")
            print("=================================================================")
            print(category_names[i].upper() ,":  ",classification_report(Y_test[:,i], y_pred[:,i]))
            print('Accuracy Score:','%','%.2f ' %  (100.0*accuracy_score(Y_test[:,i], y_pred[:,i])))
            print("=================================================================")
        
    
    # print("Exact overall multi-output comparison: clf.score(X_test_tfidf,y_test)")
        


def save_model(model, model_filepath):
    """
    Save trained model as pickle file to model_filepath 
    
    """

    pickle.dump(model, open(model_filepath, 'wb'))




def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
