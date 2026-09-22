# 🏠 House Rent Prediction

An end-to-end ML web application that predicts monthly house rent for Indian cities using a **RandomForest** model served through a **Flask** REST API and visualised with a **Streamlit** frontend.

---

## Project Structure

```
house_rent_prediction/
├── data/
│   └── House_Rent_Dataset.csv     # raw dataset
├── models/
│   ├── rent_model.pkl             # trained model  (generated)
│   └── encoders.pkl               # label encoders (generated)
├── train_model.py                 # data preprocessing + model training
├── app.py                         # Flask REST backend
├── frontend.py                    # Streamlit frontend
├── requirements.txt
└── README.md
```

---

## Quick Start

### 1 — Install dependencies
```bash
pip install -r requirements.txt
```

### 2 — Train the model
```bash
python train_model.py
```
Expected output:
```
Test MAE  : ₹X,XXX
Test RMSE : ₹X,XXX
Test R²   : 0.XXXX
✅  Model saved  → models/rent_model.pkl
✅  Encoders saved → models/encoders.pkl
```

### 3 — Start the Flask backend
```bash
python app.py
```
Runs on **http://localhost:5000**

### 4 — Start the Streamlit frontend (new terminal)
```bash
streamlit run frontend.py
```
Opens at **http://localhost:8501**

---

## API Reference

| Method | Endpoint        | Description                        |
|--------|-----------------|------------------------------------|
| GET    | `/api/health`   | Health check                       |
| GET    | `/api/options`  | Dropdown values for the UI         |
| POST   | `/api/predict`  | Predict rent from property details |

### POST `/api/predict` — Request body
```json
{
  "bhk": 2,
  "size": 1000,
  "area_type": "Super Area",
  "city": "Mumbai",
  "furnishing_status": "Semi-Furnished",
  "tenant_preferred": "Bachelors/Family",
  "bathroom": 2,
  "floor_num": "3",
  "total_floors": 5
}
```

### Response
```json
{
  "predicted_rent": 28000,
  "range_low": 25200,
  "range_high": 30800,
  "currency": "INR"
}
```

---

## Dataset Features

| Feature           | Description                              |
|-------------------|------------------------------------------|
| BHK               | Number of bedrooms                       |
| Size              | Size in square feet                      |
| Floor             | Floor number / total floors              |
| Area Type         | Super Area / Carpet Area / Built Area    |
| City              | Kolkata, Mumbai, Bangalore, Delhi, etc.  |
| Furnishing Status | Unfurnished / Semi-Furnished / Furnished |
| Tenant Preferred  | Bachelors / Family / Both                |
| Bathroom          | Number of bathrooms                      |

---

## Tech Stack

| Layer      | Technology                  |
|------------|-----------------------------|
| ML Model   | scikit-learn RandomForest   |
| Backend    | Flask + Flask-CORS          |
| Frontend   | Streamlit + Plotly          |
| Data       | pandas, numpy               |
