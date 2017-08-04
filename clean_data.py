import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def clean_df():
    df_wine = pd.read_csv('data/winemag-data_first150k.csv')

    # replace nan country with 'na'
    df_wine['country'] = df_wine['country'].fillna(value = 'na')
    # replace nan province with country
    df_wine['province'] = df_wine['province'].fillna(value = df_wine['country'])
    # replace nan region_1 with province
    df_wine['region_1'] = df_wine['region_1'].fillna(value = df_wine['province'])
    # replace nan region_2 with region_1
    df_wine['region_2'] = df_wine['region_2'].fillna(value = df_wine['region_1'])

    # create dictionary mapping countries to continents
    df_countries = pd.read_csv('data/Countries-Continents-csv.csv')
    df_countries = df_countries[['Continent', 'Country']]
    country_dict = {}
    for n in xrange(df_countries.shape[0]):
        country_dict[df_countries.iloc[n,1]] = df_countries.iloc[n,0]

    # update countries in df_wine that are not in dictionary
    country_dict['US'] = 'North America'
    country_dict['na'] = 'na'
    country_dict['South Korea'] = 'Asia'
    country_dict['England'] = 'Europe'
    country_dict['US-France'] = 'na'

    # add continent column to df_wine
    df_wine['continent'] = [country_dict[country] for country in df_wine['country']]

    df_wine.pop('Unnamed: 0')

    # remove duplicate rows
    df_wine.drop_duplicates(inplace = True)

    # write clean df_wine to csv
    df_wine.to_csv("data/wine.csv", index=False)

def extract_descriptions_and_score():
    df_wine = pd.read_csv('data/wine.csv')
    descriptions = df_wine[['description']]
    scores = df_wine[['points']]

    descriptions.to_csv("data/descriptions.csv", index=False)
    scores.to_csv("data/scores.csv", index=False)

def create_dummy_df():
    df = pd.read_csv('data/wine.csv')
    columns = ['continent', 'country', 'designation', 'province', 'region_1', 'region_2', 'variety', 'winery']
    df2 = df[columns]
    df2 = pd.get_dummies(df2, sparse=True)
    df2.to_csv("data/dummies.csv", index=False)


def plot_points_price():
    df_wine = pd.read_csv('data/wine.csv')

    df = df_wine[df_wine['price'].notnull()]
    df = df[['price', 'points']]
    df.sort_values(by=['price', 'points'], inplace=True)

    x = np.array([])
    for p in df['price'].unique():
        x = np.hstack( ( x, np.linspace( p, p+0.99, df[df['price'] == p].shape[0] ) ) )

    y = df['points']

    xx = df['price'].unique()
    yy = df.groupby('price').mean()

    plt.plot(x,y,'b')
    plt.plot(xx,yy,'r')
    plt.show()

if __name__ == '__main__':
    sns.set_style("whitegrid")

    clean_df()
    extract_descriptions_and_score()
    df_wine = pd.read_csv('data/wine.csv')
    descriptions = pd.read_csv("data/descriptions.csv")
    scores = pd.read_csv("data/scores.csv")

    # df_wine = pd.read_csv('data/wine.csv')
    #
    # descriptions = df_wine[['description']]
    # scores = df_wine[['points']]

    # asia = df_wine[df_wine['continent'] == 'Asia']
    # europe = df_wine[df_wine['continent'] == 'Europe']
    # north_america = df_wine[df_wine['continent'] == 'North America']
    # oceania = df_wine[df_wine['continent'] == 'Oceania']
    # south_america = df_wine[df_wine['continent'] == 'South America']

    # df_wine_no_price = df_wine[df_wine['price'].isnull()]
    # df_wine_with_price = df_wine[df_wine['price'].notnull()]
    #
    # top_12_countries = list(df_wine.groupby('country')[['province']].count()\
    #                         .sort_values(by='province',ascending=False).head(12).index)

    # fig, ax = plt.subplots()
    # data = [np.array(asia['points']), np.array(europe['points'])]
    # ax.violinplot( data, showmeans = True )
    # ax.set_xticks([1,2])
    # ax.set_xticklabels(('Asia', 'Europe'))
    #
    # plt.show()
