import pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error

# -----------------------------
# Load cleaned data
# -----------------------------
df = pd.read_csv("data/ratings_clean.csv")

print("✅ Data Loaded Successfully!")

# ======================================================
# ✅ VISUAL 1 — Ratings Distribution
# ======================================================
plt.figure(figsize=(8,5))
plt.hist(df["rating"], bins=5, edgecolor='black')
plt.title("Rating Distribution")
plt.xlabel("Rating")
plt.ylabel("Count")
plt.grid(axis='y', alpha=0.3)
plt.savefig("rating_distribution.png")
print("✅ rating_distribution.png saved!")
plt.show()

# ======================================================
# ✅ VISUAL 2 — Ratings per User (Top 20)
# ======================================================
user_counts = df["userId"].value_counts().head(20)

plt.figure(figsize=(10,5))
plt.bar(user_counts.index, user_counts.values)
plt.title("Top 20 Users With Most Ratings")
plt.xlabel("User ID")
plt.ylabel("Number of Ratings")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("ratings_per_user.png")
print("✅ ratings_per_user.png saved!")
plt.show()

# ======================================================
# ✅ VISUAL 3 — Actual vs Predicted Ratings
# ======================================================
print("✅ Loading trained model...")
model, user_encoder, movie_encoder = joblib.load("models/rf_model.pkl")

# Encode features
df["user_enc"] = user_encoder.transform(df["userId"])
df["movie_enc"] = movie_encoder.transform(df["movieId"])

X = df[["user_enc", "movie_enc"]]
y = df["rating"]

# Make test split
_, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

preds = model.predict(X_test)

plt.figure(figsize=(8,5))
plt.scatter(y_test[:200], preds[:200], alpha=0.6)
plt.title("Actual vs Predicted Ratings")
plt.xlabel("Actual Rating")
plt.ylabel("Predicted Rating")
plt.grid(alpha=0.3)
plt.savefig("actual_vs_predicted.png")
print("✅ actual_vs_predicted.png saved!")
plt.show()

print("\n🎉 ALL VISUALS GENERATED SUCCESSFULLY!")
