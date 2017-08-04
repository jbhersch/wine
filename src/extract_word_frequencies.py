import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

def create_word_frequency_df():
    descriptions = np.array(pd.read_csv("../data/descriptions.csv")['description'])
    vector = CountVectorizer(stop_words = 'english')
    X = vector.fit_transform(descriptions)
    words = np.array(vector.get_feature_names())
    frequencies = np.array(X.sum(axis=0))[0]
    df = pd.DataFrame( np.vstack((words, frequencies)).T )
    df.columns = ['word', 'frequency']
    df['frequency'] = df['frequency'].apply(int)
    df.sort_values(by = 'frequency', ascending = False, inplace = True)
    df.to_csv("../data/word_frequencies.csv", index = False)


if __name__ == '__main__':
    create_word_frequency_df()
    df = pd.read_csv("../data/word_frequencies.csv")
