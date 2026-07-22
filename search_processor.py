import faiss
import numpy as np


def create_search_index(embeddings):
    """
    Create a FAISS search index from embeddings.
    """

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(dimension)

    index.add(embeddings)

    return index


def search_similar_chunks(index, question_embedding, chunks, number_of_results=1):
    """
    Search for the most similar text chunks.
    """

    question_embedding = np.array(question_embedding).astype("float32")

    distances, indexes = index.search(
        question_embedding,
        number_of_results
    )

    results = []

    for i in indexes[0]:
        results.append(chunks[i])

    return results