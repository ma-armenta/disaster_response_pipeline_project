import sys
from sqlalchemy import create_engine
import pandas as pd
import numpy as np
import nltk
nltk.download(['punkt', 'wordnet'])
nltk.download('stopwords')
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report
import re
from sklearn.svm import SVC
import pickle

def load_data(database_filepath):
    # Connect to database
    engine = create_engine('sqlite:///' + database_filepath)

    # Load data from table - Clean_Messages
    df = pd.read_sql("select * from Clean_Messages", engine)

    # separate datasets
    X = df['message']
    Y = df.drop(['id', 'message', 'original', 'genre'], axis = 1)

    # get category names
    category_names = Y.columns.values

    return X, Y, category_names

def tokenize(text):
    # All text to lowercase
    text = text.lower()

    # Remove punctuation characters
    text = re.sub(r"[^a-zA-Z0-9]", " ", text) 

    # Tokenization
    tokens = word_tokenize(text)

    # Lemmatize
    lemmatizer = WordNetLemmatizer()
    
    # Removing stop words
    tokens = [w for w in tokens if w not in stopwords.words("english")]

    # Removing any whitespaces
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():

    # Creating pipeline as indicated by Project Insutrctions
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer = tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier(n_estimators = 10,
                                                             min_samples_split = 10)))
    ])
    
    # For some reason adding parameters never worked for me. I tried this on the jupyter notebook
    # as well as on my computer.
    # This issue happens whenever I used MultiOutputClassifier as directed by the project instructions.
    # add parameters for grid search
    # parameters = {'tfidf__use_idf':[True, False],
    #               'clf__estimator__n_estimators':[10, 25], 
    #               'clf__estimator__min_samples_split':[2, 3]}
    # cv = GridSearchCV(pipeline, param_grid=parameters)

    # return cv
    return pipeline

def evaluate_model(model, X_test, Y_test, category_names):
    # Get predictions using built model
    y_pred = model.predict(X_test)

    # Evaluation using classification_report and iterating through columns as indicated in project
    for i in range(len(category_names)):
        print(category_names[i])
        print(classification_report(Y_test.iloc[:,i],y_pred[:,i]))

def save_model(model, model_filepath):
    # Saving model as a pickle file
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