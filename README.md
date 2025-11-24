# 🌧️ Rain Prediction using LSTM

This project predicts **future rainfall** using a **Long Short-Term Memory (LSTM)** deep learning model.
It follows a **fully modular, production-ready pipeline architecture**, including data ingestion, preprocessing, model training, and prediction services.

---

## 📌 Features

- End-to-end modular ML pipeline
- Data ingestion with automated file handling
- Data transformation with scaling, SMOTE, encoding & sequence generation
- LSTM-based deep learning model for rain prediction
- Training pipeline with model saving
- Prediction pipeline for real-time inference
- Custom logging and exception handling
- Clean, reproducible project structure

---

## 🧠 Project Architecture

```
Rain Prediction (LSTM)
│
├── Data Ingestion
├── Data Transformation
├── Model Trainer
├── Train Pipeline
└── Predict Pipeline
```

---

## 📂 Folder Structure

```
📦 Rain-Prediction-LSTM
├── artifacts/
│   ├── model.h5
│   ├── scaler.pkl
│   └── target_scaler.pkl
│
├── src/
│   ├── data_ingestion/
│   ├── data_transformation/
│   ├── model_trainer/
│   ├── pipeline/
│   │   ├── train_pipeline.py
│   │   └── predict_pipeline.py
│   ├── logger.py
│   ├── Exception.py
│   └── Config.py
│
├── notebook/
│   ├── EDA.ipynb
│   └── data_cleaning.ipynb
│
├── app.py
├── requirements.txt
└── README.md
```

---

## 🔧 Technologies Used

- Python
- TensorFlow / Keras (LSTM)
- NumPy, Pandas
- Scikit-learn
- Matplotlib, Seaborn
- Flask (optional for deployment)

---

## 📊 Model Overview

### LSTM Architecture:

- Input → LSTM Layer → Dense Layer → Output
- Handles time-series sequences generated in the transformation stage
- Scalers ensure consistent state between train & predict

---

## 📘 Example Prediction Output

```json
{
  "date": "2025-12-01",
  "predicted_rainfall": 22.54
}
```

---

# 👥 Contributors

We gratefully acknowledge the efforts and collaboration of our team members who contributed to the **Rain Prediction using LSTM** project:

| Name               | GitHub Profile                                                  | Contribution Summary                                                                                        |
| ------------------ | --------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------- |
| **Tejas Bagul**    | [github.com/2013Tejas](https://github.com/2013Tejas)            | Data preprocessing, feature engineering, model experimentation, and ML pipeline integration.                |
| **Suhas Kolhe**    | [github.com/suhaskolhe1](https://github.com/suhaskolhe1)        | Project architecture, modularization, pipeline development, logging/exception framework, and documentation. |
| **Shruti Patil**   | [github.com/shrutipatil](https://github.com/shrutipatil-140104) | Dataset preparation, EDA, visualizations, and reporting.                                                    |
| **Aditya Ambhore** | —                                                               | Model training support, hyperparameter tuning, and backend utilities.                                       |
| **Ujwal Khairnar** | —                                                               | API integration, testing, and deployment assistance.                                                        |

---
