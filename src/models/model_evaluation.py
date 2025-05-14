import numpy as np
import pandas as pd
import logging
import pickle
import json
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

# Logging configuration
try:
    logger = logging.getLogger('model_evaluation')
    logger.setLevel('DEBUG')

    console_handler = logging.StreamHandler()
    console_handler.setLevel('DEBUG')

    file_handler = logging.FileHandler('model_evaluation.log')
    file_handler.setLevel('ERROR')

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
except Exception as e:
    print(f"Error configuring logging: {str(e)}")
    raise

def load_model():
    try:
        logger.info("Loading trained model")
        with open('models/model.pkl', 'rb') as f:
            model = pickle.load(f)
        logger.debug("Model loaded successfully")
        return model
    except FileNotFoundError:
        logger.error("Model file not found")
        raise
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise

def load_test_data():
    try:
        logger.info("Loading test data")
        test_data = pd.read_csv('./data/features/test_bow.csv')
        
        X_test = test_data.iloc[:,0:-1].values
        y_test = test_data.iloc[:,-1].values
        
        logger.debug(f"Test data loaded successfully. Shape: {X_test.shape}")
        return X_test, y_test
    except FileNotFoundError:
        logger.error("Test data file not found")
        raise
    except Exception as e:
        logger.error(f"Error loading test data: {str(e)}")
        raise

def evaluate_model(model, X_test, y_test):
    try:
        logger.info("Evaluating model performance")
        
        # Make predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        metrics_dict = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'auc': roc_auc_score(y_test, y_pred_proba)
        }
        
        logger.debug(f"Evaluation metrics calculated: {metrics_dict}")
        return metrics_dict
    except Exception as e:
        logger.error(f"Error during model evaluation: {str(e)}")
        raise

def save_metrics(metrics_dict):
    try:
        logger.info("Saving evaluation metrics")
        with open('metrics.json', 'w') as f:
            json.dump(metrics_dict, f, indent=4)
        logger.info("Metrics saved successfully")
    except Exception as e:
        logger.error(f"Error saving metrics: {str(e)}")
        raise

def main():
    try:
        # Load model
        model = load_model()
        
        # Load test data
        X_test, y_test = load_test_data()
        
        # Evaluate model
        metrics = evaluate_model(model, X_test, y_test)
        
        # Save metrics
        save_metrics(metrics)
        
        logger.info("Model evaluation completed successfully")
    except Exception as e:
        logger.error(f"Fatal error in model evaluation pipeline: {str(e)}")
        raise
    finally:
        logger.info("Model evaluation process finished")

if __name__ == "__main__":
    main()