import os
import sys
import numpy as np
import pandas as pd
from dataclasses import dataclass
from sklearn.preprocessing import MinMaxScaler

from src.logger import logging
from src.Exception import CustomException
from src.Config import sequence_length

@dataclass
class DataTransformationConfig:
    scaler_path: str = os.path.join("artifacts", "scaler.pkl")
    target_scaler_path: str = os.path.join("artifacts", "target_scaler.pkl")
 


class DataTransformation:

    def __init__(self):
        self.config = DataTransformationConfig()
        self.scaler = MinMaxScaler()
        self.target_scaler = MinMaxScaler()   

    def create_sequences(self, X, y, seq_len):
        """
        Convert continuous time-series into LSTM 3D sequences.
        Example: use last 30 days to predict next day.
        """
        X_seq, y_seq = [], []

        for i in range(len(X) - seq_len):
            X_seq.append(X[i:i + seq_len])
            y_seq.append(y[i + seq_len])

        return np.array(X_seq), np.array(y_seq)

    def init_data_transformation(self, X_train, y_train, X_test, y_test):
        logging.info("DataTransformation: Transformation Started")

        try:
            
            # 1. Convert all inputs to numpy
           
            X_train = np.array(X_train)
            X_test = np.array(X_test)
            y_train = np.array(y_train).reshape(-1, 1)
            y_test = np.array(y_test).reshape(-1, 1)

            
            # IMPORTANT FIX:
            # Do NOT apply SMOTE.
            # Do NOT oversample LSTM time-series data.
            

           
            # 2. Scale features ONLY
           
            logging.info("Scaling feature data")
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)

           
            
           
            if len(np.unique(y_train)) > 2:     
                logging.info("Target looks continuous → scaling applied")
                y_train_scaled = self.target_scaler.fit_transform(y_train)
                y_test_scaled = self.target_scaler.transform(y_test)
            else:
                logging.info("Target is classification → NOT scaling 0/1 labels")
                y_train_scaled = y_train
                y_test_scaled = y_test

            
            # 4. Create sequences
            
            seq_len = sequence_length
            logging.info(f"Creating LSTM sequences (window={seq_len})")

            X_train_seq, y_train_seq = self.create_sequences(
                X_train_scaled, y_train_scaled, seq_len
            )

            X_test_seq, y_test_seq = self.create_sequences(
                X_test_scaled, y_test_scaled, seq_len
            )

            logging.info(f"LSTM Train Shape = {X_train_seq.shape}")
            logging.info(f"LSTM Test Shape  = {X_test_seq.shape}")

           
            # 5. Save Scalers
            
            import joblib
            os.makedirs("artifacts", exist_ok=True)
            joblib.dump(self.scaler, self.config.scaler_path)
            joblib.dump(self.target_scaler, self.config.target_scaler_path)

            logging.info("Scalers saved successfully")

            
            # 6. Return outputs
            
            return (
                X_train_seq, y_train_seq,
                X_test_seq, y_test_seq
            )

        except Exception as e:
            raise CustomException(e, sys)
