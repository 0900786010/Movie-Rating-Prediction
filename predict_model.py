import joblib
import sys
import pandas as pd

# Load model + encoders
model, user_encoder, movie_encoder = joblib.load("models/rf_model.pkl")

def predict_rating(user_id, movie_id):
    try:
        u = user_encoder.transform([int(user_id)])[0]
    except:
        u = -1  # unseen user

    try:
        m = movie_encoder.transform([int(movie_id)])[0]
    except:
        m = -1  # unseen movie

    pred = model.predict([[u, m]])[0]
    return pred

if __name__ == "__main__":
    user_id = int(sys.argv[1])
    movie_id = int(sys.argv[2])

    prediction = predict_rating(user_id, movie_id)
    print(f"Predicted rating for User {user_id} on Movie {movie_id}: {prediction:.2f}")
