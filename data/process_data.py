import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """Load the messages and categories data.
    Merge them and return the merged dataframe.

    Parameters
    ----------
    messages_filepath: String
        Path to the messages csv file
    generator: String
        Path to the categories csv file

    Returns
    -------
    Pandas Dataframe
        Messages df merged with categories df
    """
    messages_df = pd.read_csv(messages_filepath)
    categories_df = pd.read_csv(categories_filepath)

    return messages_df.merge(categories_df, on='id')


def clean_data(df):
    """Clean and prepare the merged messages and categories data.

    Parameters
    ----------
    df: Pandas Dataframe
        Pandas Dataframe with the messages and categories data

    Returns
    -------
    Pandas Dataframe
        Cleaned data
    """
    categories = _getting_categories(df)

    df.drop('categories', axis=1, inplace=True)
    df = pd.concat([df, categories], axis=1)
    df.drop_duplicates(inplace=True)

    related_mode = df['related'].mode()[0]
    df.loc[df['related'] == 2, 'related'] = related_mode

    return df


def save_data(df, database_filename):
    """Save the data on the database

    Parameters
    ----------
    df: Pandas Dataframe
        Pandas Dataframe with the cleaned data
    database_filename: String
        The database filename
    """
    engine = create_engine(f'sqlite:///{database_filename}')
    df.to_sql('disasters_df', engine, index=False, if_exists='replace')


def _getting_categories(df):
    """Gets the formatted categories

    Parameters
    ----------
    df: Pandas Dataframe
        Pandas Dataframe with the messages and categories data

    Returns
    -------
    Pandas Dataframe
        Dataframe with the cleaned categories
    """
    categories = df.categories.str.split(';', expand=True)
    row = categories.iloc[0, :]

    category_colnames = row.apply(lambda x: x.split('-')[0]).values
    categories.columns = category_colnames

    for column in categories:
        categories[column] = \
            categories[column].astype(str).str.split('-').str[-1]

        categories[column] = pd.to_numeric(categories[column])

    return categories


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = \
            sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)

        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)

        print('Cleaned data saved to database!')

    else:
        print('Please provide the filepaths of the messages and categories '
              'datasets as the first and second argument respectively, as '
              'well as the filepath of the database to save the cleaned data '
              'to as the third argument. \n\nExample: python process_data.py '
              'disaster_messages.csv disaster_categories.csv '
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
