import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn import metrics
import pickle_io
from wineregressor import WineRegressor

def load_regressor(model_path = "../models/ridge_model.pkl",
                   vectorizer_path = "../vectorizers/ridge_vectorizer.pkl"):
    return WineRegressor(model_path, vectorizer_path)

if __name__ == '__main__':
    descriptions = np.array(pd.read_csv("../data/descriptions.csv")['description'])
    scores = np.array(pd.read_csv("../data/scores.csv")['points'])

    tfidf = TfidfVectorizer(ngram_range = (1,2))
    X = tfidf.fit_transform(descriptions)

    X_train, X_test, y_train, y_test = train_test_split(X, scores, test_size=0.2, random_state=42)

    alpha = 0.05
    print "alpha = {0}".format(alpha)
    ridge_regression = Ridge(alpha=alpha).fit(X_train, y_train)

    y_train_pred = ridge_regression.predict(X_train)
    y_test_pred = ridge_regression.predict(X_test)
    #
    print metrics.mean_squared_error(y_train, y_train_pred)
    # # 0.0242984200951 (with dupes)
    # 0.0383813947369
    print metrics.mean_squared_error(y_test, y_test_pred)
    # # 1.51264915328 (with dupes)
    # 3.10895709535
    # pickle_io.pickle_save("../models/ridge_model.pkl", ridge_regression)
    # pickle_io.pickle_save("../vectorizers/ridge_vectorizer.pkl", tfidf)

    # wine_regressor = load_regressor()
