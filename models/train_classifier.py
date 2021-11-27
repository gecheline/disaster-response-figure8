import sys
import pandas as pd
import pickle

# import libraries
import numpy as np
from sqlalchemy import create_engine

from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

from classifier import SequentialClassifier
from tokenizer import tokenize

    
def load_data(database_filepath):
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql('SELECT * FROM Messages', engine)
    X = df['message']
    Y = df.drop(columns=['id', 'message', 'original', 'genre'])
    
    rel2_inds = list(np.squeeze(np.argwhere(Y['related'].values==2)))
    Y = Y.drop(rel2_inds, axis=0)
    X = X.drop(rel2_inds, axis=0)
    return X, Y, Y.columns


def build_model():
    
    pipeline_bow = Pipeline([
    ('count_vect', CountVectorizer(tokenizer=tokenize)),
    ('tfidf', TfidfTransformer())
    ])
    
    sequential_clf = SequentialClassifier(clf_related = LogisticRegression(max_iter=500, 
                                                                       class_weight='balanced',
                                                                       C = 1.0, 
                                                                       solver = 'lbfgs'), 
                                      clf_type = MultiOutputClassifier(LogisticRegression(max_iter=500, 
                                                                       class_weight='balanced',
                                                                       C = 1.0, 
                                                                       solver = 'lbfgs')),
                                      clf_aid = MultiOutputClassifier(LogisticRegression(max_iter=500, 
                                                                       class_weight='balanced',
                                                                       C = 2.0, 
                                                                       solver = 'lbfgs')), 
                                      clf_weather = MultiOutputClassifier(LogisticRegression(max_iter=500, 
                                                                       class_weight='balanced',
                                                                       C = 1.0, 
                                                                       solver = 'newton-cg')), 
                                      clf_infrastructure = MultiOutputClassifier(LogisticRegression(max_iter=500, 
                                                                       class_weight='balanced',
                                                                       C = 0.5, 
                                                                       solver = 'newton-cg')))
    pipeline = Pipeline([
        ('text_transform', pipeline_bow),
        ('clf', sequential_clf)
    ])
    
    return pipeline


def evaluate_model(model, X_test, Y_test, category_names):
    y_pred = model.predict(X_test)
    for i,category in enumerate(category_names):
        print(category, classification_report(Y_test.values[:,i], y_pred[:,i]))


def save_model(model, model_filepath):
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