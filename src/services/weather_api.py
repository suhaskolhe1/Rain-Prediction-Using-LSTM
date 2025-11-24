# src/service/api_service.py

import requests
import pandas as pd

from src.logger import logging
from src.Exception import CustomException
from src.pipeline.predict_pipeline import PredictPipeline
from datetime import datetime

class WeatherAPIService:

    def __init__(self, api_key, base_url):
        self.api_key = api_key
        self.base_url = base_url
        self.pipeline = PredictPipeline()

    def fetch_weather(self, location):
        """
        Fetch daily weather data for last 30 days.
        """
        try:
            url = f"{self.base_url}{location}/last30days?unitGroup=metric&key={self.api_key}"

            logging.info(f"Fetching weather API: {url}")
            res = requests.get(url)

            #if res.status_code != 200:
             #  raise Exception(f"API Error {res.status_code}: {res.text}")

            data =pd.read_json("E:/IDE/Python/PythonProject/Rain/Rain-Prediction-Using-LSTM/data/interim/jalgaon.json")


            days = []

          
            for d in data["days"][:30]:

                # Convert preciptype to 0 or 1
                precip_raw = d.get("preciptype")

                if isinstance(precip_raw, list) and len(precip_raw) > 0:
                    precip_string = precip_raw[0]
                else:
                    precip_string = "none"

                preciptype_val = 1 if precip_string == "rain" else 0

                # Parse date → day, month, year
                dt = datetime.strptime(d["datetime"], "%Y-%m-%d")

                days.append({
                    "tempmax": d.get("tempmax", 0),
                    "tempmin": d.get("tempmin", 0),
                    "temp": d.get("temp", 0),
                    "dew": d.get("dew", 0),
                    "humidity": d.get("humidity", 0),
                    "precipprob": d.get("precipprob", 0),
                    "precipcover": d.get("precipcover", 0),
                    "preciptype": preciptype_val,                  
                    "windgust": d.get("windgust", 0),
                    "windspeed": d.get("windspeed", 0),
                    "winddir": d.get("winddir", 0),
                    "sealevelpressure": d.get("pressure", 0),      
                    "cloudcover": d.get("cloudcover", 0),
                    "visibility": d.get("visibility", 0),
                    "day": dt.day,
                    "month": dt.month,
                    "year": dt.year,
                   # "precip": d.get("precip", 0)                  
                })

            df = pd.DataFrame(days)

            logging.info("API data converted to DataFrame (finalized)")

            return (df,data)


        except Exception as e:
            raise CustomException(e)

    def get_rain_prediction(self, location):
        """
        Fetch → Convert → Predict
        """
        try:
            df,data = self.fetch_weather(location)
            prediction = self.pipeline.predict(df)

            result = {
                "location": location,
                "rain_prediction": prediction,
                "data":data
            }

            logging.info("Final rain prediction computed")
            return result

        except Exception as e:
            raise CustomException(e)
