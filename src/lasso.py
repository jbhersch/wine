import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn import metrics
from wineregressor import WineRegressor

def load_regressor(model_path = "../models/lasso_alpha_1.pkl",
                   vectorizer_path = "../vectorizers/lasso_alpha_1_vectorizer.pkl"):
    return WineRegressor(model_path, vectorizer_path)

if __name__ == '__main__':
    descriptions = np.array(pd.read_csv("../data/descriptions.csv")['description'])
    scores = np.array(pd.read_csv("../data/scores.csv")['points'])

    tfidf = TfidfVectorizer(ngram_range = (1,2))
    X = tfidf.fit_transform(descriptions)

    X_train, X_test, y_train, y_test = train_test_split(X, scores, test_size=0.2, random_state=42)

    alpha = 1
    lasso_regression = Lasso(alpha=alpha).fit(X_train, y_train)

    y_train_pred = lasso_regression.predict(X_train)
    y_test_pred = lasso_regression.predict(X_test)

    print metrics.mean_squared_error(y_train, y_train_pred)
    # Out[3]: 10.369675424413952

    print metrics.mean_squared_error(y_test, y_test_pred)
    # Out[4]: 10.440012273733897
