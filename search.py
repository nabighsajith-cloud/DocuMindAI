from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np


# 1. Read the PDF
pdf_path = "sample.pdf"
reader = PdfReader(pdf_path)

full_text = ""

for page in reader.pages:
    text = page.extract_text()
    full_text += text + "\n"


# 2. Split text into smaller chunks
chunk_size = 300
chunks = []

for i in range(0, len(full_text), chunk_size):
    chunk = full_text[i:i + chunk_size]
    chunks.append(chunk)


# 3. Load the embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")


# 4. Convert chunks into embeddings
chunk_embeddings = model.encode(chunks)

chunk_embeddings = np.array(chunk_embeddings).astype("float32")


# 5. Create FAISS search index
dimension = chunk_embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)

index.add(chunk_embeddings)


# 6. Ask a question
question = "How many projects must each student complete?"

question_embedding = model.encode([question])

question_embedding = np.array(question_embedding).astype("float32")


# 7. Search for the most relevant chunk
distances, indexes = index.search(question_embedding, k=1)


# 8. Display the result
print("\nQUESTION:")
print(question)

print("\nMOST RELEVANT INFORMATION:\n")

for i in indexes[0]:
    print(chunks[i])