"""
Data loading and preprocessing utilities.
"""
import pandas as pd


def load_data(filepath):
    """Load dataset from file."""
    # TODO: Implement data loading
    df = pd.read_csv(filepath)
    return df


def preprocess(df):
    """Clean and preprocess the data."""
    # TODO: Implement preprocessing steps
    # e.g., handle missing values, normalize, encode categoricals
    return df


def split_data(df, test_size=0.2, random_state=42):
    """Split data into train and test sets."""
    from sklearn.model_selection import train_test_split
    # TODO: Define your features and target
    # X = df.drop(columns=['target'])
    # y = df['target']
    # return train_test_split(X, y, test_size=test_size, random_state=random_state)
    pass
