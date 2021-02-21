import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request
from plotly.graph_objs import Bar, Histogram, Box
import joblib
from sqlalchemy import create_engine
import sys
sys.path.insert(1, '../models/')


app = Flask(__name__)


def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('disasters_df', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():

    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

    labels_list, num_obs_labels = _get_num_obs_per_label(df)

    df['words_count'] = df['message'].apply(lambda x: len(word_tokenize(x)))
    words_count_list = \
        list(df.words_count)

    words_count_list_filtered = list(df[df['words_count'] < 61].words_count)

    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
        {
            'data': [
                Bar(
                    x=num_obs_labels,
                    y=labels_list,
                    orientation='h'
                )
            ],

            'layout': {
                'title': 'Number of observations per label',
                'xaxis': {
                    'title': "Label"
                }
            }
        },
        {
            'data': [
                Histogram(
                    x=words_count_list
                )
            ],

            'layout': {
                'title': 'Histogram number of words per message',
                'xaxis': {
                    'title': "Number of words"
                }
            }
        },
        {
            'data': [
                Box(
                    x=words_count_list
                )
            ],

            'layout': {
                'title': 'Box Plot number of words per message'
            }
        },
        {
            'data': [
                Histogram(
                    x=words_count_list_filtered,
                    xbins=dict(size=2),
                    autobinx=False
                    )
            ],

            'layout': {
                'title': 'Histogram number of words per message filtering only'
                ' messages with less then 61 words',
                'xaxis': {
                    'title': "Number of words"
                }
            }
        }
    ]

    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


def _get_num_obs_per_label(df):
    """Gets the number of observations per each label

    Parameters
    ----------
    df: Pandas Dataframe
        Pandas Dataframe with the messages and categories data

    Returns
    -------
    Tuple
        The labels
        The number of observations for each label
    """

    cols_df = pd.DataFrame(columns=['col', 'num_obs'])
    for col in df.iloc[:, 4:].columns:
        cols_df = \
            cols_df.append({'col': col,
                            'num_obs': df[df[col] == 1].shape[0]},
                           ignore_index=True)

    cols_df = cols_df.sort_values(['num_obs'])

    return cols_df['col'].values, cols_df['num_obs'].values

# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '')

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file.
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=8080, debug=True)


if __name__ == '__main__':
    main()
