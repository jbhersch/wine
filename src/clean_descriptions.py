import pandas as pd
import numpy as np
from string import printable

def remove_unprintable(corpus):
    clean_corpus = []
    for doc in corpus:
        new_doc = []
        for letter in doc:
            if letter in printable:
                new_doc.append(letter)
        clean_corpus.append(str(''.join(new_doc)))
    return clean_corpus

def clean_descriptions():
    descriptions = np.array(pd.read_csv("../data/descriptions.csv")['description'])
    df = pd.DataFrame(remove_unprintable(descriptions))
    df.columns = ['description']
    df.to_csv("../data/descriptions.csv", index=False)


if __name__ == '__main__':
    descriptions = np.array(pd.read_csv("../data/descriptions.csv")['description'])
    scores = np.array(pd.read_csv("../data/scores.csv")['points'])
