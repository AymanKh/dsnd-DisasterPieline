# import libraries
import sys
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
from scipy.stats import hmean
import os
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from nltk.tokenize import word_tokenize
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import make_scorer, accuracy_score, f1_score, fbeta_score, classification_report
from nltk.stem import WordNetLemmatizer
from scipy.stats.mstats import gmean
from sqlalchemy import create_engine
import re
import nltk

nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])


def load_data(database_filepath):
    """
    a function to load the data file
    
    INPUTS:
        database_filepath: path to SQLite db
    OUTPUTS:
        X: feature DataFrame
        Y: label DataFrame
        categorical_names: used for visualization
    """
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql_table(database_filepath,engine)
    X = df['message']
    Y = df.iloc[:,4:]
    categorical_names = Y.columns
    return X, Y, categorical_names


def tokenize(text):
    """
    a function to tokenize the provided text
    
    INPUTS:
        text: the text that needs to be tokenized
    OUTPUTS:
        tokens: tokenized text ready for processing
    """

    # use nltk library 
    tokens = word_tokenize(text)

    # use nltk lemmatizer 
    lemma = WordNetLemmatizer()
    
    # lemmatize, make it lower case, and remove spaces
    tokens = [lemma.lemmatize(t).lower().strip() for t in tokens]

    return tokens
    


def build_model():
    """
    build the ML model and specify a pipeline
    
    INPUTS:
        NONE
    OUTPUTS:
        Pipeline: the ML pipeline
    """
    pipeline = Pipeline([('vect', CountVectorizer(tokenizer = tokenize)),
                      ('tfidf', TfidfTransformer()),
                      ('classifier', MultiOutputClassifier(RandomForestClassifier(**params)))])
    return pipeline

def evaluate_model(model, X_test, Y_test, category_names):
    """
    Evaluate the models' performance and print it on the console  
    INPUTS:
        model object, data, and category names
    OUTPUTS:
        NONE
    """

    # get predicted categories 
    predicted = model.predict(X_test)

    # print accuracy, percisison, recall and f1-score
    print(classification_report(Y_test, predicted, target_names=category_names))

    # print accuracy for every category
    for i in range(36):
        print("Category {} -> Accuracy Score - {}".format(category_names[i], accuracy_score(Y_test[:, i], predicted[:, i])))


def save_model(model, model_filepath):
    """
    serialize the model and save it locally 
    
    INPUTS:
        model: the actual model object
        model_filepath: the desired destiniation
    OUTPUTS:
        NONE
    """
    pickle.dump(model, open(model_filepath, "wb"))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
#         print(X)
#         print(Y)
#         print(category_names)
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