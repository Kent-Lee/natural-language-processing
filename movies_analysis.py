import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report


def parse_ombd(omdb):
    # filter out movies with no plot
    omdb = omdb[omdb['omdb_plot'] != 'N/A']
    omdb = omdb.reset_index()
    omdb = omdb.drop(columns='index')
    return omdb


def parse_wikidata(wikidata):
    # filter out movies with no publication date 
    wikidata = wikidata[wikidata['publication_date'].notnull()].copy()
    # parse year from format "yyyy-mm-dd" and convert it to int to compare
    wikidata[['year', 'month', 'date']] = wikidata['publication_date'].str.split('-', expand=True)
    wikidata['year'] = wikidata['year'].astype(int)
    wikidata = wikidata[(wikidata['year'] < 2019) & (wikidata['year'] > 1979)]
    wikidata = wikidata.reset_index()
    wikidata = wikidata[['year', 'imdb_id']]
    return wikidata


def parse_plot(omdb):
    # 1. convert all letters to lowercase
    # 2. keep letters and digits
    #   [\w] == [A-Za-z0-9_] == match any alphabet and digit
    #   [^\w] == negate \w == match any character NOT alphabet and digit
    #   [\'-] == in case of words with ' and -
    omdb_plot = omdb['omdb_plot'].str.lower().str.replace(r'[^\w\s\'-]', '')
    return omdb_plot.tolist()


def parse_genre(omdb):
    # convert Series into list and return unique elements of that list
    genres = np.concatenate(omdb['omdb_genres'])
    return np.unique(genres)


def plot_text_length_histogram(omdb):
    counts = omdb['omdb_plot'].str.len()
    fig, ax = plt.subplots()
    counts.hist(bins=np.arange(0,5000,50), figsize=(16, 9), ax=ax)
    plt.title('Characters in Plots')
    plt.xlabel('Characters', fontsize=12)
    plt.ylabel('Plots', fontsize=12)
    ax.set_axisbelow(True)
    ax.grid(linestyle=':')
    plt.savefig('output/text_in_plot.png')
    plt.show()


def plot_genre_vs_occurrence(genres, count):
    counts = pd.DataFrame({'genre':genres, 'count':np.sum(count, axis=0)})
    fig, ax = plt.subplots()
    counts.plot(x='genre', y='count', kind='bar', legend=False, figsize=(16, 9), ax=ax)
    ax.set_axisbelow(True)
    ax.grid(linestyle=':')
    plt.title('Number of Occurrences of each Genre')
    plt.ylabel('Occurrences', fontsize=12)
    plt.xlabel('Genres', fontsize=12)
    plt.xticks(rotation=35, ha='right')
    plt.savefig('output/genre_vs_occurrence.png')
    plt.show()


def plot_genre_trend(genre_trend):
    fig, ax = plt.subplots()
    genre_trend.plot(figsize=(16, 9), ax=ax)
    ax.set_axisbelow(True)
    ax.grid(linestyle=':')
    plt.title('Trends in Movie Popularity for Different Genres over Time')
    plt.ylabel('Movies', fontsize=12)
    plt.xlabel('Year', fontsize=12)
    plt.legend(title='Genres')
    plt.savefig('output/genre_trend.png')
    plt.show()


def plot_genre_correlation(correlation):
    fig, ax = plt.subplots()
    correlation.plot(kind='bar', legend=False, figsize=(16, 9), ax=ax)
    ax.set_axisbelow(True)
    ax.grid(linestyle=':')
    plt.title('Correlation between Genre and Year')
    plt.ylabel('Correlation Coefficient', fontsize=12)
    plt.xlabel('Genres', fontsize=12)
    plt.xticks(rotation=35, ha='right')
    plt.savefig('output/genre_correlation.png')
    plt.show()


def find_optimal_parameter(method ,X_train, X_test, y_train, y_test, genres):
    if method == 'NB':
        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(stop_words='english')),
            ('clf', OneVsRestClassifier(MultinomialNB(fit_prior=True, class_prior=None))),
    ])
    if method == 'SVM':
        pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(stop_words='english')),
        ('clf', OneVsRestClassifier(LinearSVC())),
    ])
    if method == 'LR':
        pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(stop_words='english')),
        ('clf', OneVsRestClassifier(LogisticRegression(solver='sag'))),
    ])
    parameters = {
        'tfidf__max_df': (0.25, 0.5, 0.75),
        'tfidf__min_df': (0.005, 0.01, 0.015),
        'tfidf__ngram_range': [(1, 1), (1, 2), (1, 3)],
        'clf__estimator__C': [0.01, 0.1, 1],
        'clf__estimator__class_weight': ['balanced', None],
    }

    # cross validate and select the best parameter configuration at the same time
    grid_search_tune = GridSearchCV(pipeline, parameters, cv=3, n_jobs=2, verbose=3)
    grid_search_tune.fit(X_train, y_train)

    best_clf = grid_search_tune.best_estimator_
    predictions = best_clf.predict(X_test)

    print(classification_report(y_test, predictions, target_names=genres))
    print()
    print("Best parameters:")
    print(grid_search_tune.best_estimator_.steps)
    print()


def print_input_prediction(mlb, X_test, predictions):
    target_names = mlb.inverse_transform(predictions)
    for item, labels in zip(X_test, target_names):
        print('{0}... => {1}\n'.format(item[0:40], ', '.join(labels)))


def main():
    omdb = pd.read_json('data/omdb-data.json.gz', orient='record', lines=True, encoding='utf-8')
    wikidata = pd.read_json('data/wikidata-movies.json.gz', orient='record', lines=True, encoding='utf-8')
    
    omdb = parse_ombd(omdb)
    genres = parse_genre(omdb)

    # index all labels - pick n labels to be 1, all other are 0
    mlb = MultiLabelBinarizer()
    y = mlb.fit_transform(omdb['omdb_genres'])
    X = parse_plot(omdb)
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    
    plot_genre_vs_occurrence(genres, y)
    plot_text_length_histogram(omdb)

    # find_optimal_parameter('NB' ,X_train, X_test, y_train, y_test, genres)
    # find_optimal_parameter('SVM' ,X_train, X_test, y_train, y_test, genres)
    # find_optimal_parameter('LR' ,X_train, X_test, y_train, y_test, genres)

    # get parameter values from tests above
    classifier = Pipeline(steps=[
        ('tfidf', TfidfVectorizer(stop_words='english', max_df=0.25, min_df=0.005, ngram_range=(1, 1))),
        ('clf', OneVsRestClassifier(LinearSVC())),
    ])

    classifier.fit(X_train, y_train)
    predictions = classifier.predict(X_test)

    # print_input_prediction(mlb, X_test, predictions)
    print('==========================================================================')
    print('Summary of Prediction Accuracy')
    print('==========================================================================')
    print(classification_report(y_test, predictions, target_names=genres))
    print('The warning means that there is no F-score to calculate for the specified\n'
        'labels, so the F-score for them are considered to be 0.0. Since we wanted\n'
        'an average of the score, we must take into account that the score of 0 was\n'
        'included in the calculation; hence scikit-learn displays the warning.\n')
    print('The reason for the missing values is that some labels in y_test (truth) do\n'
        'not appear in predictions. In other words, the labels are never predicted.')
    print('--------------------------------------------------------------------------\n')

    wikidata = parse_wikidata(wikidata)
    genre_trend = pd.merge(omdb, wikidata, on='imdb_id')
    # https://stackoverflow.com/questions/42012152/unstack-a-pandas-column-containing-lists-into-multiple-rows
    # explode list of column to rows
    genre_trend = pd.DataFrame({
        'genre': np.concatenate(genre_trend['omdb_genres']),
        'year': np.repeat(genre_trend['year'], genre_trend['omdb_genres'].str.len()),
    })
    top_ten_genres = genre_trend.groupby('genre')['year'].count().reset_index(name='count')
    top_ten_genres = top_ten_genres.sort_values('count', ascending=False).reset_index()
    top_ten_genres = top_ten_genres['genre'].tolist()[:10]

    # https://stackoverflow.com/questions/47998025/python-line-plot-for-values-grouped-by-multiple-columns
    genre_trend = genre_trend.groupby(['year', 'genre'])['genre'].count()
    genre_trend = genre_trend.unstack()
    genre_trend = genre_trend[top_ten_genres]
    plot_genre_trend(genre_trend)

    # https://stackoverflow.com/questions/34896455/how-to-do-pearson-correlation-of-selected-columns-of-a-pandas-data-frame/34896835
    # correlate a column with multiple columns
    print('==========================================================================')
    print('Correlation between Genre and Year')
    print('==========================================================================')
    genre_trend['year'] = genre_trend.index
    corr = genre_trend[genre_trend.columns[0:]].corr()['year'][:-1]
    plot_genre_correlation(corr)
    print(corr)
    print('--------------------------------------------------------------------------')


if __name__ == "__main__":
    main()