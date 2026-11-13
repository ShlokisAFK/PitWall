# PitWall F1 Race Predictor

An XGBoost-powered F1 race prediction pipeline trained on 4 seasons of historical data (2022-2025, 1836 rows). Predicts race finishing positions using engineered features from qualifying data, driver form, and circuit history. Deployed on a Raspberry Pi 5 with a FastAPI backend.

## Model Performance 2026 Season (15 races)

| Metric | Value |
|---|---|
| Podium Accuracy | 93.3% |
| Avg Position Error (MAE) | 1.06 positions |
| Training Data | 2022-2025, 1836 rows |

## Features Used

- Qualifying gap to pole position
- Grid position
- Driver form (last 3 races average finish)
- Team form (last 3 races average finish)
- DNF rate (last 3 races)
- Circuit history score
- Starting tyre compound
- Wet/dry condition flag
- Sprint race position (where applicable)
- Track type (power, street, high downforce, high deg, balanced)

## Project Structure

```
F1-Predictor/
├── train.py               XGBoost training with early stopping
├── model/
│   └── model.json         Trained model
├── api/
│   └── main.py            FastAPI backend
├── pipeline/
│   ├── fetch.py           FastF1 data fetching
│   ├── features.py        Feature engineering
│   └── post_race.py       Post-race results pipeline
├── frontend/
│   ├── index.html         Dashboard with race countdown and accuracy stats
│   └── predict.html       Race predictor interface
└── data/
    ├── features.csv        Feature matrix
    ├── raw_races.csv       Raw race results
    └── accuracy_log.json   Per-race accuracy log
```

## Running Locally

```bash
pip install fastapi uvicorn xgboost fastf1 pandas numpy scikit-learn apscheduler slowapi pydantic

# Train the model
python train.py

# Start the API
uvicorn api.main:app --host 0.0.0.0 --port 8000

# Open frontend/index.html in browser
```

## Stack

- Model: XGBoost (2000 estimators, MAE objective, early stopping)
- Backend: FastAPI
- Data: FastF1 API
- Frontend: Vanilla JS, HTML/CSS
- Deployment: Raspberry Pi 5
