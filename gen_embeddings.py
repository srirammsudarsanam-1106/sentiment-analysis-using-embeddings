from sentence_transformers import SentenceTransformer

embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)
sample = [
    "This movie was amazing",
    "Worst movie ever"
]
embeddings = embedding_model.encode(sample)
print(embeddings.shape)