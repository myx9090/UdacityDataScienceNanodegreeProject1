import sys
from collections import deque
import numpy as np
import pandas as pd
import os
from sqlalchemy import create_engine
import re

import nltk
nltk.download('punkt')
nltk.download('stopwords')

from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import *
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import *
from sklearn.metrics import *
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.externals import joblib


def load_data(database_filepath):
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql_table('P1Data', engine)
    X = 'message'
    Y = [col for col in df.columns.tolist() if col not in ['id', 'message', 'original', 'genre']]
    
    df.drop(['child_alone'], axis=1, inplace=True)

    index = df[df['related']==2].index
    print(len(index))
    df.drop(index, inplace=True)

    Y.remove('child_alone')
    
    return df[X], df[Y], Y


def tokenize(text):
    processed_text = re.sub(r"[^A-Za-z0-9]", " ", text.lower()) # remove special characters.
    tokens = nltk.tokenize.word_tokenize(processed_text) # into word tokens
    
    stemmer = nltk.stem.PorterStemmer()
    stemmed_words = [stemmer.stem(_word) for _word in tokens if _word not in nltk.corpus.stopwords.words("english")]
    
    return stemmed_words


def build_model():
    pipeline = Pipeline([
        ("tfidf_vectorizer", TfidfVectorizer(tokenizer=tokenize)) , 
        ("clf", MultiOutputClassifier(RidgeClassifier(random_state=2020, alpha=0.1)))
    ])
    
    return pipeline

def evaluate_model(model, X_test, Y_test, category_names):
    accuracy, precision, recall, f1, index = deque(), deque(), deque(), deque(), deque() # placeholder for metrics
    
    preds = model.predict(X_test)
    
    for i, col in enumerate(category_names):
        accuracy.append(accuracy_score(Y_test[col], preds[:, i]))
        precision.append(precision_score(Y_test[col], preds[:, i]))
        recall.append(recall_score(Y_test[col], preds[:, i]))
        f1.append(f1_score(Y_test[col], preds[:, i]))
        index.append(col)
        
    result = pd.DataFrame({
        'accuracy' : accuracy ,
        'precision' : precision , 
        'recall' : recall , 
        'f1' : f1} , 
        index = index
    )
    
    print(result)


def save_model(model, model_filepath):
    joblib.dump(model, model_filepath)

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