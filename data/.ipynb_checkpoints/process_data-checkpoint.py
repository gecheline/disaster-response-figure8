import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    '''
    Loads the messages and categories data from user-provided filepaths.
    
    Parameters
    ----------
    messages_filepath: str
        Path to the file containing the messages data (.csv).
    categories_filepath: str
        Path to the file containing the categories data (.csv).
        
    Returns
    -------
    df: pandas.DataFrame instance
        A dataframe containing the concatenated messages and categories data.
    '''
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    return pd.concat([messages, categories], axis=1)
    

def clean_data(df):
    '''
    Cleans the data by transforming the categories column in separate columns in one-hot notation.
    
    The 'categories' column is replaced by N columns, split up by ';' delimiter in the first row.
    The full tag in each column is replaced with the last character, typically a 1 or 0, resulting
    in a one-hot transformation of the categories data.
    
    Parameters
    ----------
    df: pandas.DataFrame instance
        A dataframe containing the messages data and categories column.
    
    Returns
    -------
    df: pandas.DataFrame instance
        Cleaned dataframe.
    '''

    categories_expanded = df['categories'].str.split(pat=';', expand=True)
    categories_expanded.head()
    row = categories_expanded.iloc[0,:]
    category_colnames = [name.split('-')[0] for name in row]
    categories_expanded.columns = category_colnames
    for column in categories_expanded:
    # set each value to be the last character of the string
        categories_expanded[column] = categories_expanded[column].str.split('-', expand=True).drop(0, axis=1)
        categories_expanded[column] = categories_expanded[column].astype(int)
    
    df = df.drop(columns=['categories'])
    df = pd.concat([df, categories_expanded], axis=1)
    return df.drop_duplicates().drop(columns=['child_alone'])


def save_data(df, database_filename):
    '''Saves the cleaned dataframe to a sql database.
    
    Parameters
    ----------
    database_filename: str
        Path to the file to save the database.
    
    '''

    engine = create_engine('sqlite:///{}'.format(database_filename))
    df.to_sql('Messages', engine, index=False)


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()