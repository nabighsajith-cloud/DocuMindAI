from pdf_processor import extract_text_from_pdf, split_text_into_chunks
from embedding_processor import create_embeddings
from search_processor import create_search_index, search_similar_chunks


# 1. Read the PDF
pdf_path = "sample.pdf"

text = extract_text_from_pdf(pdf_path)


# 2. Split the text into chunks
chunks = split_text_into_chunks(text)

print("Total chunks:", len(chunks))


# 3. Create embeddings for all chunks
embeddings = create_embeddings(chunks)

print("Embeddings created successfully")


# 4. Create FAISS search index
index = create_search_index(embeddings)

print("Search index created successfully")


# 5. Ask a question
question = "How many projects must each student complete?"


# 6. Convert the question into an embedding
question_embedding = create_embeddings([question])


# 7. Search for the most relevant chunk
results = search_similar_chunks(
    index,
    question_embedding,
    chunks
)


# 8. Display the result
print("\nQUESTION:")
print(question)

print("\nMOST RELEVANT INFORMATION:\n")

for result in results:
    print(result)