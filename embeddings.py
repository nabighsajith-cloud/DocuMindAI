from sentence_transformers import SentenceTransformer

# Load the AI embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Example text
texts = [
    "Each student must complete two projects.",
    "The internship includes a Generative AI project."
]

# Convert text into embeddings
embeddings = model.encode(texts)

print(embeddings)