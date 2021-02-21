import sys
import nltk
import re
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from sqlalchemy import create_engine
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
import joblib
from xgboost import XGBClassifier
import extractor_transformers
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger', 'stopwords'])


def load_data(database_filepath):
    """Load the data with the messages and the category

    Parameters
    ----------
    database_filepath: String
        Filepath to the database

    Returns
    -------
    Tuple
        Features np.array
        Targets np.array
        Categories names list
    """
    engine = create_engine(f'sqlite:///{database_filepath}')
    df = pd.read_sql_table('disasters_df', engine)
    X = df.message.values
    Y = df.iloc[:, 4:].values
    category_names = df.iloc[:, 4:].columns

    return X, Y, list(category_names)


def tokenize(text):
    """Tokenize the text

    Parameters
    ----------
    text: String
        The message to be tokenized

    Returns
    -------
    List
        List with the clean tokens
    """
    text = text.lower()
    text = re.sub("[^a-zA-Z0-9]", " ", text)
    tokens = word_tokenize(text)
    tokens = [w for w in tokens if w not in stopwords.words('english')]

    lemmatizer = WordNetLemmatizer()
    stemmer = PorterStemmer()

    clean_tokens_list = []
    for tok in tokens:
        lemmatizer_tok = lemmatizer.lemmatize(tok).strip()
        clean_tok = stemmer.stem(lemmatizer_tok)
        clean_tokens_list.append(clean_tok)

    return clean_tokens_list


def build_model():
    """Build the model

    Returns
    -------
    sklearn.pipeline.Pipeline
        The model
    """
    pipeline = Pipeline([
        ('features', FeatureUnion([

            ('text_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer())
            ])),

            ('num_verbs', extractor_transformers.CountPosTagTransformer('VB')),
            ('num_nouns', extractor_transformers.CountPosTagTransformer('JJ')),
            ('num_adjectives',
             extractor_transformers.CountPosTagTransformer('PRP')),
            ('num_pronouns',
             extractor_transformers.CountPosTagTransformer('NN'))
        ])),

        ('clf', MultiOutputClassifier(XGBClassifier(eval_metric="logloss")))
    ])

    parameters = {
        'clf__estimator__learning_rate': [0.1, 0.01],
    }

    cv_pipeline = GridSearchCV(pipeline, param_grid=parameters)

    return cv_pipeline


def evaluate_model(model, X_test, Y_test, category_names):
    """Get the evaluation metrics for each category

    Parameters
    ----------
    model: sklearn.pipeline.Pipeline
        The model to be evaluated
    X_test: Np.array
        Numpy array with the test features
    Y_test: Np.arary
        Numpy array with the test targets
    category_names: List
        List with the categories
    """
    y_predicted = model.predict(X_test)
    for i, col in enumerate(category_names):
        print(col)
        print(classification_report(Y_test[:, i],
                                    y_predicted[:, i]))


def save_model(model, model_filepath):
    """Saves the model at a pickle file

    Parameters
    ----------
    model: sklearn.pipeline.Pipeline
        The model to be saved
    model_filepath: String
        The model filepath
    """
    joblib.dump(model, model_filepath)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = \
            train_test_split(X, Y, test_size=0.2)

        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, Y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '
              'as the first argument and the filepath of the pickle file to '
              'save the model to as the second argument. \n\nExample: python '
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
