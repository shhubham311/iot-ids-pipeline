import pandas as pd
import os
import logging

# Setup basic logging to see what's happening
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# -----------------------
# Get the directory where this script is located (src/)
# Go up one level to get the Project Root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Construct the path to the raw data
DEFAULT_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "raw", "RT_IOT2022")
# -----------------------

def load_data(data_path: str = None) -> pd.DataFrame:
    """
    Loads the RT-IoT2022 dataset from a CSV file.
    Defaults to the project's data/raw folder if no path is provided.
    """
    if data_path is None:
        data_path = DEFAULT_DATA_PATH
        
    try:
        logging.info(f"Loading data from {data_path}")
        # The dataset usually has headers, if not, add header=None
        df = pd.read_csv(data_path)
        logging.info(f"Data loaded successfully. Shape: {df.shape}")
        return df
    except FileNotFoundError:
        logging.error(f"File not found at: {data_path}")
        raise

def basic_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    """
    Performs basic cleaning:
    1. Drop duplicates.
    2. Handle missing values (simple imputation for now).
    """
    logging.info("Starting basic cleaning...")
    
    # 1. Drop duplicates
    initial_rows = len(df)
    df = df.drop_duplicates()
    if len(df) < initial_rows:
        logging.info(f"Dropped {initial_rows - len(df)} duplicate rows.")
    
    # 2. Handle Missing Values
    # Strategy: If numerical, fill with median. If categorical, fill with mode.
    # This is a basic baseline strategy.
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            if df[col].dtype == 'object':
                mode_val = df[col].mode()[0]
                df[col].fillna(mode_val, inplace=True)
            else:
                median_val = df[col].median()
                df[col].fillna(median_val, inplace=True)
                
    logging.info("Cleaning completed.")
    return df

if __name__ == "__main__":
    # This block runs only when we execute this file directly.
    # Useful for quick testing of the functions.
    
    df = load_data() 
    df_clean = basic_cleaning(df)
    print(df_clean.head())