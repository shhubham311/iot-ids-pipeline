import sys
import os
import joblib
import mlflow
import mlflow.xgboost
import pandas as pd
import xgboost as xgb
from sklearn.metrics import classification_report, f1_score, accuracy_score
import logging

# Add parent directory to path to import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocess import preprocess_and_split
from src.data import load_data, basic_cleaning

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def train_model(data_path: str, experiment_name: str = "IoT_IDS_Experiment"):
    """
    1. Preprocesses data
    2. Trains XGBoost Classifier
    3. Evaluates on Test set
    4. Logs everything to MLflow
    """
    
    # Set the MLflow Experiment
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run():
        logging.info("Starting training run...")

        # 1. Preprocess
        # This function drops unknown, encodes, and splits
        X_train, X_test, y_train, y_test, feature_encoders, target_encoder = preprocess_and_split(data_path)
        
        os.makedirs("../models", exist_ok=True)
        joblib.dump(feature_encoders, "../models/feature_encoders.pkl")
        joblib.dump(target_encoder, "../models/target_encoder.pkl")
        logging.info("Feature and Target encoders saved to ../models/")
        
        # Get the number of classes dynamically
        num_classes = len(target_encoder.classes_)
        logging.info(f"Number of Classes detected: {num_classes}")
        logging.info(f"Classes: {target_encoder.classes_}")

        # 2. Define XGBoost Model
        # We use use_label_encoder=False because XGBoost wants to handle labels internally now
        model = xgb.XGBClassifier(
            objective='multi:softmax',  # Multiclass classification
            num_class=num_classes,      # Tell it how many attack types exist
            eval_metric='mlogloss',     # Metric for training loss
            #use_label_encoder=False,
            n_estimators=100,           # Number of trees
            max_depth=6,                # Depth of trees
            learning_rate=0.1,
            random_state=42,
            n_jobs=-1                   # Use all CPU cores
        )

        # 3. Log Parameters
        mlflow.log_params({
            "n_estimators": 100,
            "max_depth": 6,
            "learning_rate": 0.1,
            "num_classes": num_classes
        })

        # 4. Train
        logging.info("Training XGBoost model...")
        model.fit(X_train, y_train, verbose=False)
        logging.info("Training complete.")

        # 5. Predict & Evaluate
        y_pred = model.predict(X_test)

        # Calculate Metrics
        acc = accuracy_score(y_test, y_pred)
        f1_macro = f1_score(y_test, y_pred, average='macro') # Crucial for imbalance
        
        # Log Metrics
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_macro", f1_macro)
        
        logging.info(f"Test Accuracy: {acc:.4f}")
        logging.info(f"Test F1 Macro (Score for rare attacks): {f1_macro:.4f}")

        # Print Classification Report (Precision/Recall per class)
        # This helps us see WHICH specific attack type the model is failing on
        report = classification_report(y_test, y_pred, target_names=target_encoder.classes_)
        logging.info("\nClassification Report:\n" + report)
        
        # Save report as an artifact in MLflow so we can view it later
        with open("classification_report.txt", "w") as f:
            f.write(report)
        mlflow.log_artifact("classification_report.txt")

        # 6. Log the Model
        # This saves the binary model file to MLflow tracking server
        mlflow.xgboost.log_model(model, name="model") 
        # Or explicitly: mlflow.xgboost.log_model(model, name="model")
        
        model_path = os.path.join("../models", "xgboost_model.pkl")
        joblib.dump(model, model_path)
        logging.info(f"Model saved locally to {model_path}")
        
        logging.info("Model and metrics logged to MLflow successfully.")
        
if __name__ == "__main__":
    # UPDATE THIS PATH to match your specific location
    DATA_PATH = "/Users/shubh/Work/iot_intrusion_detection/data/raw/RT_IOT2022"
    
    train_model(DATA_PATH)