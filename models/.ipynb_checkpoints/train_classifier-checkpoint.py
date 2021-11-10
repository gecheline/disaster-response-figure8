import sys
import pandas as pd
import pickle

# import libraries
import numpy as np
from sqlalchemy import create_engine

from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer
from nltk.stem.wordnet import WordNetLemmatizer

from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier

from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

class SequentialClassifier(BaseEstimator, ClassifierMixin):

    
    def __init__(self, clf_related, 
                 clf_type, 
                 clf_aid,
                 clf_weather, 
                 clf_infrastructure,
                ):
        
        # Initialize each classifier with their provided kwargs
        self.clf_related = clf_related
        self.clf_type = clf_type
        self.clf_aid = clf_aid
        self.clf_weather = clf_weather
        self.clf_infrastructure = clf_infrastructure
        
    
        self.all_columns = ['related', 'request', 'offer', 'aid_related', 'medical_help',
                           'medical_products', 'search_and_rescue', 'security', 'military',
                           'water', 'food', 'shelter', 'clothing', 'money',
                           'missing_people', 'refugees', 'death', 'other_aid',
                           'infrastructure_related', 'transport', 'buildings', 'electricity',
                           'tools', 'hospitals', 'shops', 'aid_centers', 'other_infrastructure',
                           'weather_related', 'floods', 'storm', 'fire', 'earthquake', 'cold',
                           'other_weather', 'direct_report']
        self.type_columns = ['aid_related', 'weather_related', 'infrastructure_related', 
                             'request', 'offer', 'direct_report']
        self.aid_columns = ['medical_help',
                           'medical_products', 'search_and_rescue', 'security', 'military',
                           'water', 'food', 'shelter', 'clothing', 'money',
                           'missing_people', 'refugees', 'death', 'other_aid']
        self.weather_columns = ['floods', 'storm', 'fire', 'earthquake', 'cold','other_weather']
        self.infrastructure_columns = ['transport', 'buildings', 'electricity',
                                       'tools', 'hospitals', 'shops', 'aid_centers', 'other_infrastructure']
        

    def fit(self, X, Y):
        
        Y1 = Y['related']
        
        self.clf_related.fit(X, Y1)
        
        related_mask = Y['related']==1
        self.clf_type.fit(X[related_mask], Y[related_mask][self.type_columns])
        
        aid_mask = Y['aid_related']==1
        self.clf_aid.fit(X[aid_mask], Y[aid_mask][self.aid_columns])
        
        weather_mask = Y['weather_related']==1
        self.clf_weather.fit(X[weather_mask], Y[weather_mask][self.weather_columns])
        
        infrastructure_mask = Y['infrastructure_related']==1
        self.clf_infrastructure.fit(X[infrastructure_mask], Y[infrastructure_mask][self.infrastructure_columns])
        
        return self
    
    def predict(self, X):
        
        #this is where the sequential part comes!
        y_predict = pd.DataFrame(np.zeros((X.shape[0], len(self.all_columns))), 
                                 columns= self.all_columns)
        
        y_predict['related'] = self.clf_related.predict(X)
        
        related_mask = y_predict['related'] == 1        
        y_types = self.clf_type.predict(X[related_mask])
        
        y_predict.loc[related_mask, 'aid_related'] = y_types[:,0]
        y_predict.loc[related_mask, 'weather_related'] = y_types[:,1]
        y_predict.loc[related_mask, 'infrastructure_related'] = y_types[:,2]
        y_predict.loc[related_mask, 'request'] = y_types[:,3]
        y_predict.loc[related_mask, 'offer'] = y_types[:,4]
        y_predict.loc[related_mask, 'direct_report'] = y_types[:,5]
        
        aid_mask = y_predict['aid_related'] == 1
        weather_mask = y_predict['weather_related'] == 1
        infrastructure_mask = y_predict['infrastructure_related'] == 1
        
        y_aid_types = self.clf_aid.predict(X[aid_mask])
        y_weather_types = self.clf_weather.predict(X[weather_mask])
        y_infrastructure_types = self.clf_infrastructure.predict(X[infrastructure_mask])
        
        for i,col in enumerate(self.aid_columns):
            y_predict.loc[aid_mask, col] = y_aid_types[:,i]
        for i,col in enumerate(self.weather_columns):
            y_predict.loc[weather_mask, col] = y_weather_types[:,i]
        for i,col in enumerate(self.infrastructure_columns):
            y_predict.loc[infrastructure_mask, col] = y_infrastructure_types[:,i]
            
        return y_predict.values
    

def load_data(database_filepath):
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql('SELECT * FROM Messages', engine)
    X = df['message']
    Y = df.drop(columns=['id', 'message', 'original', 'genre'])
    
    rel2_inds = list(np.squeeze(np.argwhere(Y['related'].values==2)))
    Y = Y.drop(rel2_inds, axis=0)
    X = X.drop(rel2_inds, axis=0)
    return X, Y, Y.columns

def remove_stopwords(words):
    return [word for word in words if word not in stopwords.words('english')]
    
def lemmatize(words):
    words = [WordNetLemmatizer().lemmatize(word, pos='n') for word in words]
    words = [WordNetLemmatizer().lemmatize(word, pos='v') for word in words]
    words = [WordNetLemmatizer().lemmatize(word, pos='a') for word in words]
    return words

def tokenize_twitter(text):
    from nltk.tokenize import TweetTokenizer
    return TweetTokenizer().tokenize(text)

def tokenize(text):
    return remove_stopwords(lemmatize(tokenize_twitter(text)))


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