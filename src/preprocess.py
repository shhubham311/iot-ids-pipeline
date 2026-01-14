import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib # Used to save the encoders for later use
import logging
import sys

# Add the parent directory to path so we can import data.py
sys.path.append('..')
from src.data import load_data, basic_cleaning

logging.basicConfig(level=logging.INFO)

def preprocess_and_split(data_path: str, test_size: float = 0.2, random_state: int = 42):
    """
    1. Loads data
    2. Drops useless columns
    3. Encodes categorical features and target
    4. Splits into Train/Test sets
    5. Returns X_train, X_test, y_train, y_test, and the encoders (to save later)
    """
    
    # 1. Load & Clean
    df = load_data(data_path)
    df = basic_cleaning(df)
    
    # 2. Drop 'Unnamed: 0' (It's just an index, not a feature)
    if 'Unnamed: 0' in df.columns:
        df.drop('Unnamed: 0', axis=1, inplace=True)
        logging.info("Dropped 'Unnamed: 0' column.")

    # 3. Separate Features (X) and Target (y)
    X = df.drop('Attack_type', axis=1)
    y = df['Attack_type']

    # 4. Encode Categorical Features
    # We identified 'proto' and 'service' as categorical objects.
    cat_cols = ['proto', 'service']
    
    feature_encoders = {} # Dictionary to store our encoders
    
    for col in cat_cols:
        if col in X.columns:
            le = LabelEncoder()
            # fit_transform learns the categories and converts to numbers
            X[col] = le.fit_transform(X[col].astype(str)) 
            feature_encoders[f'le_{col}'] = le
            logging.info(f"Encoded column '{col}' with {len(le.classes_)} classes.")
            
    # 5. Encode Target Label (Attack_type)
    target_encoder = LabelEncoder()
    y_encoded = target_encoder.fit_transform(y.astype(str))
    logging.info(f"Encoded Target with classes: {target_encoder.classes_}")

    # 6. Train/Test Split
    # Stratify = y ensures that the rare attack types are present in both train and test sets proportionally.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=test_size, random_state=random_state, stratify=y_encoded
    )

    logging.info(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    
    return X_train, X_test, y_train, y_test, feature_encoders, target_encoder

if __name__ == "__main__":
    # Test the script
    # NOTE: Adjust the path below to match your exact file location
    DATA_PATH = "../data/raw/RT_IOT2022" 
    
    try:
        X_train, X_test, y_train, y_test, f_enc, t_enc = preprocess_and_split(DATA_PATH)
        
        print("\n--- Sample of Processed X_train ---")
        print(X_train.head())
        
        print("\n--- Sample of Encoded y_train ---")
        print(y_train[:5])
        print(f"Decoded labels: {t_enc.inverse_transform(y_train[:5])}")
        
    except Exception as e:
        print(f"Error: {e}")