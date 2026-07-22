from pdf_processor import extract_text_from_pdf, split_text_into_chunks


# Use any PDF file
pdf_path = "sample.pdf"


# Extract text from the PDF
text = extract_text_from_pdf(pdf_path)


# Split the text into chunks
chunks = split_text_into_chunks(text)


# Display the result
print("Total chunks:", len(chunks))

print("\nFirst chunk:\n")
print(chunks[0])