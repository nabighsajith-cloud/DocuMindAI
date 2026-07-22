from sentence_transformers import SentenceTransformer
import numpy as np


# Load the embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")


def create_embeddings(chunks):
    """
    Convert text chunks into numerical embeddings.
    """

    embeddings = model.encode(chunks)

    embeddings = np.array(embeddings).astype("float32")

    return embeddings