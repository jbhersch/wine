from pickle_io import pickle_load

class WineRegressor(object):
    def __init__(self, model_path, vectorizer_path):
        self.model = pickle_load(model_path)
        self.vectorizer = pickle_load(vectorizer_path)

    def transform(self, text):
        return self.vectorizer.transform(text)

    def inverse_transform(self, vector):
        return self.vectorizer.inverse_transform(vector)

    def predict_vector(self, vector):
        return self.model.predict(vector)

    def predict_text(self, text):
        return self.predict_vector(self.transform(text))

if __name__ == '__main__':
    model_path = "../models/linear_regression_model.pkl"
    vectorizer_path = "../vectorizers/linear_regression_vectorizer.pkl"

    wine_regressor = WineRegressor(model_path, vectorizer_path)
