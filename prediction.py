# predict.py

import joblib
from sentence_transformers import SentenceTransformer

classifier = joblib.load(
    "models/sentiment.pkl"
)

embedder = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

review = input("Enter review: ")

embedding = embedder.encode([review])

prediction = classifier.predict(
    embedding
)

if prediction[0] == 1:
    print("Positive")
else:
    print("Negative")