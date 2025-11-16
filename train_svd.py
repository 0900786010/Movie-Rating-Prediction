from surprise import Dataset, Reader, SVD, accuracy
from surprise.model_selection import train_test_split
import pandas as pd
import joblib
import os

# Load the clean CSV data
df = pd.read_csv("data/ratings_clean.csv")

# Surprise library needs a Reader with rating scale
reader = Reader(rating_scale=(1, 5))

# Load the dataframe for Surprise
data = Dataset.load_from_df(df[["userId", "movieId", "rating"]], reader)

# Split dataset
trainset, testset = train_test_split(data, test_size=0.2, random_state=42)

# Create SVD model
model = SVD()

print("✅ Training model... please wait")

# Train model
model.fit(trainset)

# Evaluate
predictions = model.test(testset)
rmse = accuracy.rmse(predictions)
mae = accuracy.mae(predictions)

# Create models folder
os.makedirs("models", exist_ok=True)

# Save model
joblib.dump(model, "models/svd_model.pkl")

print("\n✅ Training complete!")
print("✅ RMSE:", rmse)
print("✅ MAE:", mae)
print("✅ Model saved as: models/svd_model.pkl")
