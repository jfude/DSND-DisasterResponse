import sys
import pandas as pd
import numpy as np
import sqlalchemy as sa

def load_data(messages_filepath, categories_filepath):
    """
    Load disaster_messages and disaster_categories csv files
    and merge based on id of message

    Keyword Arguments: 
    messages_filepath:   Filepath including file name to disaster_messages.csv
    categories_filepath: Filepath including file name to disaster_categories.csv

    Returns: 
    df:  Dataframe of merged messages and categories dataframes
    """
    
    messages   = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = pd.merge(messages,categories,on = 'id', how="inner")
    return df


def clean_data(df):
    """ 
    Clean data such that each of 36 categories is a column and each row consists
    of a disaster message, corresponding id, and a one(zero) in a categorical column if the message
    does (does not) falls into a given category.

    
    Keyword Arguments: 
    df: The merged messages/categories dataframe returned from load_data()
    
    Returns:
    df: Cleaned dataframe

    """
    
    # Remove duplicates
    df.drop_duplicates(inplace=True)

    # Split the values in the categories column on the ; character so that each value becomes a separate column.
    categories = df.categories.str.split(";",expand=True)
    
    # Select the first row of the categories dataframe
    # and use this row to extract a list of new column names for categories.
    row = categories.loc[0]
    category_colnames = list(row.apply(lambda x: x[:-2]))

    
    categories.columns = category_colnames

    
    # set each value to be the last character of the stringm which is a 0 or 1
    for column in categories:
        categories[column] = categories[column].astype(str).apply(lambda x: x[-1:]).astype(float)
    

    # drop the original categories column from `df`
    df.drop(['categories'],axis=1,inplace=True)

    # Column bind df with messages with categories dataframe (which has corresponding 1 or 0 if
    # message does or does not correspond to category)
    df = pd.concat([df,categories],axis=1)

    
    # check number of duplicates again
    #np.any(df.duplicated())
    return df
    

def save_data(df, database_filename):
    """
    Save cleaned disaster messages/categories dataframe to local SQL Lite database
    using relative path

    Keyword Arguments: 
    df: dataframe to save
    database_filename: Filename of database of course
    
    """

    engine = sa.create_engine('sqlite:///' + database_filename)
    df.to_sql('DisasterResponse', engine, index=False)


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
