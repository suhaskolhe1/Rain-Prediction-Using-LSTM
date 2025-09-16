from dataclasses import dataclass
import sys
import os
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Input, LSTM, Dropout, Dense # type: ignore
from tensorflow.keras.callbacks import EarlyStopping # type: ignore

from src.Config import *
from src.Exception import CustomException


from sklearn.metrics import accuracy_score, classification_report



@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifacts","model.pkl")



class ModelTrainer:
    def __init__(self):
        self.config = ModelTrainerConfig()
    
    def build_lstm_model(self,input_shape):
        model = Sequential([
            Input(shape=input_shape), 
            LSTM(128, return_sequences=True),
            Dropout(0.3),
            LSTM(128, return_sequences=True),
            Dropout(0.2),
            LSTM(32),
            Dropout(0.2),
            Dense(1, activation='sigmoid') 
            ])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        return model

    def initiate_model_trainer(self, X, y, X_test, y_test):
        try:
            
          model = self.build_lstm_model(input_shape=(SEQUENCE_LENGTH,7))
          model.summary()
          early_stop = EarlyStopping(monitor='val_loss', patience=PATIENCE, restore_best_weights=True)
          history = model.fit(
               X, y,
               validation_split=VALIDATION_SPLIT,
               epochs=EPOCHS,
               batch_size=BATCH_SIZE,
               callbacks=[early_stop],  
               verbose=1)
          
          y_pred = (model.predict(X_test, verbose=0) > 0.5).astype(int)
          print("\nModel Evaluation:")
          print("Accuracy:", accuracy_score(y_test, y_pred))
          print("\nClassification Report:")
          print(classification_report(y_test, y_pred, zero_division=0))

        except Exception as e:
            raise CustomException(e, sys)
        