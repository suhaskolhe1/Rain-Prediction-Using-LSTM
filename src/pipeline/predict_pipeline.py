# src/pipeline/predict_pipeline.py

import os
import numpy as np
import joblib
from tensorflow.keras.models import load_model   # type: ignore

from src.logger import logging
from src.Exception import CustomException


class PredictPipeline:

    def __init__(self):
        try:
            self.model_path = os.path.join("artifacts", "model.h5")
            self.scaler_path = os.path.join("artifacts", "scaler.pkl")

            logging.info("Loading model and scaler...")
            self.model = load_model(self.model_path)
            self.scaler = joblib.load(self.scaler_path)

        except Exception as e:
            raise CustomException(e)

    def prepare_sequence(self, df, seq_len=30):
        """
        df: dataframe containing last N days weather data
        Output: numpy array (1, seq_len, features)
        """
        try:
            logging.info("Scaling and creating sequence for prediction")

            scaled = self.scaler.transform(df)
            seq = scaled[-seq_len:]  # last 30 days

            return np.array(seq).reshape(1, seq_len, df.shape[1])

        except Exception as e:
            raise CustomException(e)

    def predict(self, daily_df):
        """
        daily_df: dataframe of recent daily weather (temp, humidity, wind, etc.)
        """
        try:
            seq = self.prepare_sequence(daily_df)

            logging.info("Running model prediction...")
            pred = self.model.predict(seq)

            return float(pred[0][0])  # return  prediction

        except Exception as e:
            raise CustomException(e)
