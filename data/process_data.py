# import required libraries & packages
import sys
import pandas as pd
from sqlalchemy import create_engine # to save the clean dataset into an sqlite database


def load_data(messages_filepath, categories_filepath):
    """
    Loads and merges the messages and categories datasets.

    Parameters:
    messages_filepath (str): The file path to the CSV file containing the messages.
    categories_filepath (str): The file path to the CSV file containing the categories.

    Returns:
    pd.DataFrame: A merged DataFrame containing messages and categories.
    """
    # read in data files (two csv files)
    messages = pd.read_csv(messages_filepath)  # load messages dataset
    categories = pd.read_csv(categories_filepath)  # load categories dataset
    df = pd.merge(messages, categories, left_on='id', right_on='id', how='inner')  # merge datasets on id

    return df


def clean_data(df):
    """
    Cleans the merged DataFrame by processing the categories.

    Parameters:
    df (pd.DataFrame): The merged DataFrame containing messages and categories.

    Returns:
    pd.DataFrame: The cleaned DataFrame.
    """
    # clean data
    categories = df["categories"].str.split(';', expand=True)  # Split the 'categories' column into separate columns
    categories.columns = categories.iloc[0].str[:-2].values  # Extract new column names directly from the first row
    categories = categories[1:]  # Drop the first row since it was used for column names

    for column in categories:
        categories[column] = pd.to_numeric(categories[column].str[-1], errors='coerce').fillna(0)  # convert column from string to numeric

    df.drop(['categories'], axis=1, inplace=True)  # drop the original categories column from `df`
    df = pd.concat([df, categories], axis=1)  # concatenate the original dataframe with the new `categories` dataframe

    df = df.drop_duplicates()  # remove duplicates
    df.fillna(0, inplace=True)  # replace all NaN values with 0
    
    return df


def save_data(df, database_filename):
    '''
    Save cleaned df to database from clean_data.

    Input:
    df: Cleaned pandas DataFrame from load_data function
    database_filename: Custom defined filename for database

    Output:
    None
    '''
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('DisasterResponses', engine, index=False, if_exists='replace')
      

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

    