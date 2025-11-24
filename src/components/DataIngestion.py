from dataclasses import dataclass
import os
import pandas as pd
from src.logger import logging
from src.Exception import CustomException
import sys


@dataclass
class DataIngestionConfig:
    dataset_path: str = os.path.join("data", "interim", "cleaned.csv")
    train_path: str = os.path.join("data","processed", "train.csv")
    test_path: str = os.path.join("data","processed", "test.csv")


class DataIngestion:

    def __init__(self):
        self.config = DataIngestionConfig()

    def init_data_ingestion(self):
        logging.info("DataIngestion: Data Ingestion Started")

        try:
            
            # 1. Load dataset
            
            df = pd.read_csv(self.config.dataset_path)
            logging.info(f"Data Loaded Successfully. Shape = {df.shape}")

            
            # 2. Sort by date (Important for LSTM)
           
            if {"year", "month", "day"}.issubset(df.columns):
                df = df.sort_values(["year", "month", "day"]).reset_index(drop=True)
                logging.info("Dataset Sorted by Year-Month-Day")

            else:
                logging.warning("Date columns missing: year, month, day")

            
            # 3. Define  target
            
            
            target_column = "precip"  

            if target_column not in df.columns:
                raise CustomException(f"Target column '{target_column}' not found", sys)

            X = df.drop(columns=[target_column])
            y = df[target_column]

            
            # 4. Time-Series Split (NO SHUFFLE)
            
            split_idx = int(len(df) * 0.8)

            X_train, y_train = X.iloc[:split_idx], y.iloc[:split_idx]
            X_test, y_test = X.iloc[split_idx:], y.iloc[split_idx:]

            logging.info("Time-Series Train-Test Split Completed")
            logging.info(f"Train Shape: {X_train.shape}, Test Shape: {X_test.shape}")

            
            # 5. Save Split Data
            
            os.makedirs("artifacts", exist_ok=True)

            train_df = pd.concat([X_train, y_train], axis=1)
            test_df = pd.concat([X_test, y_test], axis=1)

            train_df.to_csv(self.config.train_path, index=False)
            test_df.to_csv(self.config.test_path, index=False)

            logging.info("Train & Test Files Saved Successfully")

            return (X_train, y_train, X_test, y_test)

        except Exception as e:
            raise CustomException(e, sys)

