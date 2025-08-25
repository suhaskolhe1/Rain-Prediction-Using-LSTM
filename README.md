
# Rain Prediction with LSTM (Flask + Pipelines)

## Quickstart
```bash
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python run.py
```

## Workflow
1. Put your CSVs (2010–2025) into `data/raw/` (same columns as Jalgaon file).
2. Configure hyperparameters in `src/config.py`.
3. Train model: `python -m src.pipelines.train_pipeline`
4. Evaluate: `python -m src.pipelines.evaluate_pipeline`
5. Run Flask app for interactive predictions: `python run.py` then open http://127.0.0.1:5000

## Notes
- Default task = classification (Rain vs No Rain) using `precip > 0` as label.
- Sequences: window of past N days to predict rain on next day.
