import os
import sys
import numpy as np
from dataclasses import dataclass
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import LSTM, Dense, Dropout # type: ignore
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau # type: ignore

from src.logger import logging
from src.Exception import CustomException
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
import matplotlib.pyplot as plt

@dataclass
class ModelTrainerConfig:
    model_path: str = os.path.join("artifacts", "model.h5")
    epochs: int = 50
    batch_size: int = 32
    lstm_units: int = 64


class ModelTrainer:

    def __init__(self):
        self.config = ModelTrainerConfig()

    def build_lstm_model(self, input_shape):
        """
        Builds LSTM architecture.
        """
        model = Sequential([
            LSTM(self.config.lstm_units, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),

            LSTM(self.config.lstm_units, return_sequences=False),
            Dropout(0.2),
            #Dense(32):-Helps combine patterns extracted by LSTM
            Dense(32, activation='relu'),
            #Dense(1):-single continuous output
            Dense(1)  
        ])

        model.compile(
            optimizer="adam",
            loss="mse",             
            metrics=["mae"]
        )

        logging.info("Model Architecture Created")
        model.summary(print_fn=lambda x: logging.info(x))
        return model

    def initiate_model_training(self, X_train, y_train, X_test, y_test):
        logging.info("Model Training Started")

        try:
            
            # 1. Build Model
            
            input_shape = (X_train.shape[1], X_train.shape[2])  # (timesteps, features)
            model = self.build_lstm_model(input_shape)

            
            # 2. Callbacks
          
            os.makedirs("artifacts", exist_ok=True)

            checkpoint = ModelCheckpoint(
                filepath=self.config.model_path,
                save_best_only=True,
                monitor="val_loss",
                mode="min",
                verbose=1
            )

            early_stop = EarlyStopping(
                monitor="val_loss",
                patience=10,
                restore_best_weights=True
            )

            reduce_lr = ReduceLROnPlateau(
                monitor="val_loss",
                patience=5,
                factor=0.5,
                min_lr=1e-6,
                verbose=1
            )

           
            # 3. Train Model
            
            history = model.fit(
                X_train, y_train,
                validation_data=(X_test, y_test),
                epochs=self.config.epochs,
                batch_size=self.config.batch_size,
                callbacks=[checkpoint, early_stop, reduce_lr],
                verbose=1
            )
    


            logging.info("Model Training Completed Successfully")

            
            # 4. Save final model 
          
            model.save(self.config.model_path)
            logging.info(f"Model Saved at {self.config.model_path}")

            return history, model

        except Exception as e:
            raise CustomException(e, sys)
