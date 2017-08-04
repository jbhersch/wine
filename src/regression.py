import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from wineregressor import WineRegressor

def load_regressor(model_path = "../models/linear_regression_model.pkl",
                   vectorizer_path = "../vectorizers/linear_regression_vectorizer.pkl"):
    return WineRegressor(model_path, vectorizer_path)



if __name__ == '__main__':
    descriptions = np.array(pd.read_csv("../data/descriptions.csv")['description'])
    scores = np.array(pd.read_csv("../data/scores.csv")['points'])

    tfidf = TfidfVectorizer(ngram_range = (1,2))
    X = tfidf.fit_transform(descriptions)

    X_train, X_test, y_train, y_test = train_test_split(X, scores, test_size=0.2, random_state=42)

    linear_regression = LinearRegression().fit(X_train, y_train)

    y_train_pred = linear_regression.predict(X_train)
    y_test_pred = linear_regression.predict(X_test)
    #
    print metrics.mean_squared_error(y_train, y_train_pred)
    # # Out[42]: 0.00032161708518821901 (with dupes)
    # 7.02606029647e-05
    # #
    print metrics.mean_squared_error(y_test, y_test_pred)
    # # Out[43]: 1.7136204479449486 (with dupes)
    # 3.61957196469
    # # wine_regressor = load_regressor()
