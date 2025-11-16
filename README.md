# 🎬 Movie Rating Prediction (Machine Learning Project)

This project predicts **how a user might rate a movie** they have not seen yet.  
It uses the **MovieLens 100k** dataset and a **Random Forest Regression model** to make rating predictions.

The project includes:
✅ Data Preprocessing  
✅ Model Training  
✅ Model Evaluation  
✅ Making Predictions  

---

## 📌 Project Features

- Load and clean MovieLens 100k dataset  
- Convert userId and movieId into numeric values  
- Train a Random Forest Regressor model  
- Evaluate the model using RMSE and MAE  
- Predict ratings for any user–movie pair  

---

## 📂 Folder Structure
movie-rating-prediction/
│
├── data/
│ ├── ml-100k/
│ │ └── u.data
│ └── ratings_clean.csv
│
├── models/
│ └── rf_model.pkl
│
├── data_prep.py
├── train_model.py
├── predict_model.py
└── README.md

---

## 🛠️ Technologies Used

- Python  
- Pandas  
- Scikit-Learn  
- Random Forest  
- Joblib  

---

# ✅ How to Run This Project

### 1️⃣ Create Virtual Environment
python -m venv venv
venv\Scripts\activate

### 2️⃣ Install Requirements
pip install pandas scikit-learn joblib

### 3️⃣ Prepare Dataset
Place the downloaded MovieLens folder here:
data/ml-100k/u.data
Run data prep:
python data_prep.py
✅ This creates: `data/ratings_clean.csv`

---

### 4️⃣ Train the Model
python train_model.py

✅ This creates: `data/ratings_clean.csv`

---

### 4️⃣ Train the Model
python train_model.py
This will:
- Train Random Forest model  
- Print RMSE & MAE  
- Save `rf_model.pkl` inside `models/`  

---

### 5️⃣ Make a Prediction
Use this format:
python predict_model.py <userId> <movieId>
Example:
python predict_model.py 10 50
Example Output:
Predicted rating for User 10 on Movie 50: 4.55

---

# 📊 Model Evaluation

Example performance (your numbers may differ):

- **RMSE:** 0.95  
- **MAE:** 0.75  

Lower values ✅ = better accuracy.

---

# ✅ Why Random Forest?

The Surprise library requires heavy C++ tools on Windows.  
Random Forest works perfectly without extra installations and gives good accuracy for recommendation tasks.

---

# 🚀 Future Improvements
- Add user and movie features (genre, age, etc.)  
- Try more advanced models (XGBoost, LightGBM)  
- Build a simple front-end to input user & movie and view predictions  

---
## 📊 Visualizations

### ⭐ Rating Distribution
This chart shows how ratings are spread across all movies.
![Rating Distribution](rating_distribution.png)

### ⭐ Ratings Per User (Top 20 Users)
This chart shows the top 20 users who rated the most movies.
![Ratings Per User](ratings_per_user.png)

### ⭐ Actual vs Predicted Ratings
This plot compares true ratings vs model predictions to show model accuracy.
![Actual vs Predicted](actual_vs_predicted.png)

# 👤 Author
**Umer Raza**


