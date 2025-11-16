import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib
import os

# Load the clean dataset
df = pd.read_csv("data/ratings_clean.csv")

# Encode userId and movieId into numbers
user_encoder = LabelEncoder()
movie_encoder = LabelEncoder()

df["user_enc"] = user_encoder.fit_transform(df["userId"])
df["movie_enc"] = movie_encoder.fit_transform(df["movieId"])

# Features and target
X = df[["user_enc", "movie_enc"]]
y = df["rating"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model
model = RandomForestRegressor(n_estimators=100, random_state=42)

print("✅ Training model...")
model.fit(X_train, y_train)

# Evaluation
pred = model.predict(X_test)
mse = mean_squared_error(y_test, pred)
rmse = mse ** 0.5
mae = mean_absolute_error(y_test, pred)


# Save model + encoders
os.makedirs("models", exist_ok=True)
joblib.dump((model, user_encoder, movie_encoder), "models/rf_model.pkl")

print("\n✅ Training complete!")
print("✅ RMSE:", rmse)
print("✅ MAE:", mae)
print("✅ Model saved: models/rf_model.pkl")
