import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import pickle


class SentimentPrediction:

    def read_data(self):
        # Read in the data
        df = pd.read_csv('../input/amazon-reviews-unlocked-mobile-phones/Amazon_Unlocked_Mobile.csv')

        # Drop missing values
        df.dropna(inplace=True)

        # Remove any 'neutral' ratings equal to 3
        df = df[df['Rating'] != 3]

        # Encode 4s and 5s as 1 (rated positively)
        # Encode 1s and 2s as 0 (rated poorly)
        df['Positively Rated'] = np.where(df['Rating'] > 3, 1, 0)

        return df

    def train(self, X_train, y_train):            
        # extracting 1-grams and 2-grams
        self.vect = TfidfVectorizer(min_df=5, ngram_range=(1,2)).fit(X_train)

        X_train_vectorized = self.vect.transform(X_train)

        X_train_vectorized = self.vect.transform(X_train)

        self.model = LogisticRegression(solver='saga')
        self.model.fit(X_train_vectorized, y_train)


    def predict(self, X_test):
        self.model.predict(self.vect.transform(X_test))
        predictions = sentiment.predict(X_test)

        print('AUC: ', roc_auc_score(y_test, self.model.decision_function(self.vect.transform(X_test))))

  
    def save(self, path):
        pickle.dump(self.model, open(path + '/SentimentPrediction.pkl','wb'))
        pickle.dump(self.vect, open(path + '/vector.pkl','wb'))

    def load(self, path):
        self.model = pickle.load(open(path + '/SentimentPrediction.pkl', 'rb'))
        self.vect = pickle.load(open(path + '/vector.pkl', 'rb'))
    

if __name__ == '__main__':

    sentiment = SentimentPrediction()

    df = sentiment.read_data()

    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(df['Reviews'], 
                                                        df['Positively Rated'], 
                                                        random_state=0)

    sentiment.train(X_train, y_train)

    print('AUC: ', roc_auc_score(y_test, sentiment.model.decision_function(sentiment.vect.transform(X_test))))

    sentiment.save('./model')


