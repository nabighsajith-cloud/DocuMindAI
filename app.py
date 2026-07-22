import streamlit as st

from pdf_processor import extract_text_from_pdf, split_text_into_chunks
from embedding_processor import create_embeddings
from search_processor import create_search_index, search_similar_chunks


st.set_page_config(
    page_title="DocuMind AI",
    page_icon="📄",
    layout="centered"
)


st.title("📄 DocuMind AI")

st.write("Upload a PDF and ask questions about its content.")


uploaded_file = st.file_uploader(
    "Upload your PDF",
    type=["pdf"]
)


if uploaded_file is not None:

    with open("uploaded_document.pdf", "wb") as file:
        file.write(uploaded_file.getbuffer())


    text = extract_text_from_pdf("uploaded_document.pdf")


    chunks = split_text_into_chunks(text)


    embeddings = create_embeddings(chunks)


    index = create_search_index(embeddings)


    st.success("PDF processed successfully!")

    st.write("Total text chunks:", len(chunks))


    question = st.text_input(
        "Ask a question about your PDF"
    )


    if st.button("Ask Question"):

        if question == "":
            st.warning("Please enter a question.")

        else:

            question_embedding = create_embeddings([question])


            results = search_similar_chunks(
                index,
                question_embedding,
                chunks
            )


            st.subheader("🔍 Most Relevant Information")


            for result in results:
                st.write(result)