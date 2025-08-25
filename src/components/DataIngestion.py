
from dataclasses import dataclass
import os




@dataclass
class DataIngestionConfig():
    train_data_path: str = os.path.join('data/', 'train.csv')
    test_data_path: str = os.path.join('data', 'test.csv')
    raw_data_path: str = os.path.join('data', 'data.csv')



if __name__ == "__main__":
    obj = DataIngestionConfig()
    print(obj.raw_data_path)