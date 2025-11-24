from flask import Flask, render_template,request
from src.services.weather_api import WeatherAPIService
from datetime import datetime
from dotenv import load_dotenv
import os

load_dotenv()
app = Flask(__name__, template_folder="templates")

# Initialize service once
service = WeatherAPIService(
    api_key=os.getenv("API_KEY"),
    base_url="https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/"
)


@app.route("/")
def index():
    
    location = "jalgaon"

    
    result = service.get_rain_prediction(location)

    
    rain_prediction = float(result["rain_prediction"])
    api_data = result["data"]["days"][0]          # today
    forecast = result["data"]["days"][:7]         # first 7 days forecast

    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    return render_template(
        "rainfall_prediction_website.html",
        location=location,
        rain_prediction=rain_prediction,
        api_data=api_data,
        forecast=forecast,
        timestamp=timestamp
    )



@app.route("/predict")
def predict_api():
    try:
        location = request.args.get("location", "jalgaon")
        result = service.get_rain_prediction(location)

        return {
            "location": location,
            "rain_prediction": float(result["rain_prediction"]),
            "api_data": result["data"]["days"][0],
            "forecast": list(result["data"]["days"])
        }

    except Exception as e:
        return {"error": str(e)}, 500


if __name__ == "__main__":
    app.run(debug=True)
