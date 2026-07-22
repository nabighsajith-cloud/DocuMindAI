from pypdf import PdfReader

pdf_path = "sample.pdf"

reader = PdfReader(pdf_path)

full_text = ""

for page in reader.pages:
    text = page.extract_text()
    full_text += text + "\n"

# Split the document into chunks
chunk_size = 500

chunks = []

for i in range(0, len(full_text), chunk_size):
    chunk = full_text[i:i + chunk_size]
    chunks.append(chunk)

# Display the chunks
for number, chunk in enumerate(chunks, start=1):
    print(f"\n--- CHUNK {number} ---\n")
    print(chunk)