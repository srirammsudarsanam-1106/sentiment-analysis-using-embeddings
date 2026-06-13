from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib
import os
from sklearn.metrics import classification_report
from collections import Counter

dataset = load_dataset("stanfordnlp/imdb")

train_dataset = dataset["train"].shuffle(seed=42)
test_dataset = dataset["test"].shuffle(seed=42)

train_texts = train_dataset["text"][:5000]
train_labels = train_dataset["label"][:5000]

test_texts = test_dataset["text"][:1000]
test_labels = test_dataset["label"][:1000]

print("Training labels:")
print(Counter(train_labels))

print("Test labels:")
print(Counter(test_labels))

embedder = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

X_train = embedder.encode(
    train_texts,
    show_progress_bar=True
)

X_test = embedder.encode(
    test_texts,
    show_progress_bar=True
)

classifier = LogisticRegression()

classifier.fit(
    X_train,
    train_labels
)

predictions = classifier.predict(X_test)
accuracy = accuracy_score(
    test_labels,
    predictions
)
print("Accuracy:", accuracy)
os.makedirs("models", exist_ok=True)
joblib.dump(classifier, "models/sentiment.pkl")

print(
    classification_report(
        test_labels,
        predictions
    )
)