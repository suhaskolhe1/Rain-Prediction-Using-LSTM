from dataclasses import dataclass
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler,LabelEncoder
from sklearn.impute  import SimpleImputer
from src.Exception import CustomException
import sys
from src.logger import logging
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

from src.Config import SEQUENCE_LENGTH


@dataclass 
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join("artifacts", "preprocessor.pkl")
    label_encoder_obj_file_path = os.path.join("artifacts", "wind_encoder.pkl")



class DataTransformation:
    def __init__(self):
        self.config = DataTransformationConfig()
        self.scaler = MinMaxScaler()
        self.encoder = LabelEncoder()
        self.imputer = SimpleImputer(strategy='mean')

    def create_sequences(self, X, y, seq_len=SEQUENCE_LENGTH):
        """
        Create fixed-length sequences for LSTM.
        """
        X_seq = [X[i:i + seq_len] for i in range(len(X) - seq_len)]
        y_seq = [y[i + seq_len] for i in range(len(X) - seq_len)]
        return np.array(X_seq), np.array(y_seq)
    
    def deg_to_compass(self,deg):
        dirs = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
        ix = round(deg / 45) % 8
        return dirs[ix]
    
    def initiate_data_transformation(self, train_path, test_path):
        try:
            logging.info("Reading train and test data")
            train_df = pd.read_csv(train_path)
            test_df= pd.read_csv(test_path)

            logging.info("Merging train and test for preprocessing consistency")

            full_df = pd.concat([train_df, test_df], ignore_index=True)

            features = ["tempmin","tempmax","winddir","windgust","humidity","sealevelpressure","temp"]
            full_df = pd.DataFrame({
                "tempmin" : full_df["tempmin"],
                "tempmax" :full_df["tempmax"],
                "winddir" : full_df["winddir"].apply(self.deg_to_compass),
                "windgust" : full_df["windgust"],
                "humidity" : full_df["humidity"],
                "sealevelpressure" : full_df["sealevelpressure"],
                "temp" : full_df["temp"],
                "RainTomorrow" : full_df["precipprob"].apply(lambda x: "Yes" if x > 50 else "No"),
           })
            
            full_df.ffill(inplace=True)
            full_df.bfill(inplace=True)

            full_df["winddir"] = self.encoder.fit_transform(full_df["winddir"].astype(str))

            full_df.dropna(subset=['RainTomorrow'], inplace=True)
            features = ["tempmin","tempmax","winddir","windgust","humidity","sealevelpressure","temp"]
            target = 'RainTomorrow'

            X=full_df[features]
            y=full_df[target]

            X=pd.DataFrame(self.imputer.fit_transform(X),columns=features)

            print("Original dataset shape:", y.shape)
            print("Original dataset distribution:\n", pd.Series(y).value_counts())
            smote = SMOTE(random_state=42)
            X_resampled, y_resampled = smote.fit_resample(X, y)
            print("\nResampled dataset shape:", y_resampled.shape)
            print("Resampled dataset distribution:\n", pd.Series(y_resampled).value_counts())

            X_scaled = self.scaler.fit_transform(X_resampled)

            X_seq, y_seq = self.create_sequences(X_scaled, y_resampled.values)

            X_train, X_test, y_train, y_test = train_test_split(X_seq, y_seq, test_size=0.2, random_state=42, shuffle=True, stratify=y_seq)
            return X_train, y_train, X_test, y_test



        except Exception as e:
            raise CustomException(e, sys)



