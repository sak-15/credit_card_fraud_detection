import pandas as pd
import ast  


# Function to get columns with missing values
def get_missing_values_columns(df):
    missing_values = df.isnull().sum()
    return missing_values[missing_values>0]

# Preprocessing function for handling null values, removing duplicates and fixing datatypes
def preprocessing(df):
    # Handling null values
    missing_values = get_missing_values_columns(df)
    if 'TX_AMOUNT' in missing_values.index:
        df['TX_AMOUNT'] = df['TX_AMOUNT'].fillna(df['mean_amount'])
    
    # fixing datetime column datatype
    df['TX_DATETIME'] = pd.to_datetime(df['TX_DATETIME'])   

    # Removing duplicates 
    df = df.drop_duplicates()

    # fixing available_terminals datatype
    df['available_terminals'] = df['available_terminals'].apply(ast.literal_eval) 

    return df

