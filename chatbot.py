from pdf_processor import extract_text_from_pdf, split_text_into_chunks
from embedding_processor import create_embeddings
from search_processor import create_search_index, search_similar_chunks


# 1. Ask for the PDF path
pdf_path = input("Enter the PDF file name: ")


# 2. Read the PDF
text = extract_text_from_pdf(pdf_path)


# 3. Split the PDF text into chunks
chunks = split_text_into_chunks(text)

print("\nPDF loaded successfully!")
print("Total chunks:", len(chunks))


# 4. Create embeddings for the PDF chunks
embeddings = create_embeddings(chunks)


# 5. Create the FAISS search index
index = create_search_index(embeddings)

print("PDF is ready for questions!")


# 6. Start the chatbot
while True:

    question = input("\nAsk a question (type 'exit' to stop): ")

    if question.lower() == "exit":
        print("Chatbot closed.")
        break

    # Convert question into an embedding
    question_embedding = create_embeddings([question])

    # Search the PDF
    results = search_similar_chunks(
        index,
        question_embedding,
        chunks
    )

    # Display the relevant information
    print("\nRelevant information:\n")

    for result in results:
        print(result)