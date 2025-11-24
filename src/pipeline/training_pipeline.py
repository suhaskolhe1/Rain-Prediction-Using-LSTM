import sys
from src.logger import logging
from src.Exception import CustomException

from src.components.DataIngestion import DataIngestion
from src.components.DataTransformation import DataTransformation
from src.components.ModelTrainer import ModelTrainer


def start_training_pipeline():
    logging.info("============== PIPELINE STARTED ==============")

    try:
      
        # 1. DATA INGESTION
       
        ingestion = DataIngestion()
        X_train, y_train, X_test, y_test = ingestion.init_data_ingestion()

        logging.info("Data Ingestion Completed")
        logging.info(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
        logging.info(f"X_test: {X_test.shape}, y_test: {y_test.shape}")

       
        # 2. DATA TRANSFORMATION
      
        transformer = DataTransformation()
        (
            X_train_seq,
            y_train_seq,
            X_test_seq,
            y_test_seq
        ) = transformer.init_data_transformation(X_train, y_train, X_test, y_test)

        logging.info("Data Transformation Completed")
        logging.info(f"LSTM Train Shape: {X_train_seq.shape}")
        logging.info(f"LSTM Test  Shape: {X_test_seq.shape}")

    
        # 3. MODEL TRAINING
      
        trainer = ModelTrainer()
        history, model = trainer.initiate_model_training(
            X_train_seq, y_train_seq, X_test_seq, y_test_seq
        )

        logging.info("Model Training Completed Successfully")
        logging.info("============== PIPELINE FINISHED ==============")

        return model, history

    except Exception as e:
        raise CustomException(e, sys)


if __name__ == "__main__":
    start_training_pipeline()
