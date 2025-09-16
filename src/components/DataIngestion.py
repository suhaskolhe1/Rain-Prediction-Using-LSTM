
from dataclasses import dataclass
import os
import pandas as pd
from src.logger import logging
from src.Exception import CustomException
import sys
from sklearn.model_selection import train_test_split

from  src.components.DataTransformation import DataTransformation
from  src.components.ModelTrainer import ModelTrainer

@dataclass
class DataIngestionConfig():
    raw_data_dir:str = os.path.join("data","raw")
    interim_data_path: str = os.path.join("data", "interim", "merged.csv")
    train_data_path: str = os.path.join("data", "processed", "train.csv")
    test_data_path: str = os.path.join("data", "processed", "test.csv")


class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def init_data_ingestion(self):
        logging.info("DataIngestion:Data Ingestion Started")
        try:
            #Collect All Raw Files
            all_files = [
                os.path.join(self.ingestion_config.raw_data_dir,f)
                for f in os.listdir(self.ingestion_config.raw_data_dir)
                if f.endswith(".csv")
            ]

            logging.info(f"Found {len(all_files)} raw files")

            df_list=[pd.read_csv(file) for file in all_files]
            df=pd.concat(df_list,ignore_index=True)

            logging.info("DataIngestion: All CSVs merged successfully")

            ## 3. Save interim merged data
            os.makedirs(os.path.dirname(self.ingestion_config.interim_data_path),exist_ok=True)
            df.to_csv(self.ingestion_config.interim_data_path,index=False)
            
            logging.info(f"Merged data saved at {self.ingestion_config.interim_data_path}")

            logging.info("Train-test split initiated")
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)


            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)
            train_set.to_csv(self.ingestion_config.train_data_path, index=False)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False)

            logging.info("Train and test sets saved in processed folder")
            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path,
            )

        except Exception as e:
            pass
            raise CustomException(e,sys)



if __name__ == "__main__":
    obj = DataIngestion()
    train_data,test_data=obj.init_data_ingestion()

    data_transformation = DataTransformation()
    X, y, X_test, y_test = data_transformation.initiate_data_transformation(train_data, test_data)
    model = ModelTrainer()
    model.initiate_model_trainer(X, y, X_test, y_test)
