import os
import streamlit as st
from llama_index.llms.ollama import Ollama
from llama_index.core import StorageContext, KnowledgeGraphIndex
from PyPDF2 import PdfReader
import tempfile
import shutil
from llama_index.core import Document
import time


class StudyBot:
    def __init__(self):
        """Initialize StudyBot with Llama and necessary components."""
        self.llm = Ollama(
            model="llama3:instruct",
            server="http://localhost:11434/api/chat",
            timeout=3600  # Increased timeout
        )
        self.documents = []
        self.generated_text = ""
        self.query_engine = None

    def load_pdf(self, filepath):
        """Extract text from a PDF file."""
        try:
            reader = PdfReader(filepath)
            return "".join([page.extract_text() for page in reader.pages if page.extract_text()])
        except Exception as e:
            return f"Error reading {filepath}: {e}"

    def initialize(self, documents, max_retries=3):
        """Initialize the knowledge graph index with the given documents."""
        try:
            document_objects = []
            chunk_size = 5000  # Adjustable chunk size
            for doc in documents:
                chunks = [doc["text"][i:i + chunk_size] for i in range(0, len(doc["text"]), chunk_size)]
                document_objects.extend([Document(text=chunk) for chunk in chunks])

            storage_context = StorageContext.from_defaults()
            embed_model = "local:BAAI/bge-m3"

            for attempt in range(1, max_retries + 1):
                try:
                    print(f"Building KnowledgeGraphIndex (Attempt {attempt}/{max_retries})...")
                    start_time = time.time()
                    index = KnowledgeGraphIndex.from_documents(
                        documents=document_objects,
                        max_triplets_per_chunk=1,
                        storage_context=storage_context,
                        embed_model=embed_model,
                        include_embeddings=False,
                        llm=self.llm,
                        timeout=9600  # Extended timeout
                    )
                    print(f"Index built successfully in {time.time() - start_time:.2f} seconds.")
                    self.query_engine = index.as_query_engine(
                        include_text=True,
                        response_mode="tree_summarize",
                        embedding_mode="none",
                        similarity_top_k=5,
                        llm=self.llm,
                    )
                    return
                except TimeoutError:
                    print(f"Timeout during index creation. Retrying ({attempt}/{max_retries})...")
                except Exception as e:
                    print(f"Error during index creation: {e}")
                    if attempt == max_retries:
                        raise e
        except Exception as e:
            print(f"Error initializing index: {e}")

    def summarize(self, text, chunk_size=500, context_size=250):
        """Summarize the text into concise bullet points."""
        summaries = []
        try:
            for i in range(0, len(text), chunk_size):
                start_idx = max(i - context_size, 0)
                end_idx = min(i + chunk_size + context_size, len(text))
                chunk = text[start_idx:end_idx]

                prompt = f"Summarize the following text in concise bullet points:\n\n{chunk}"
                response = self.llm.complete(prompt)
                # Access the 'text' attribute directly
                summaries.append(response.text.strip())
        except TimeoutError:
            summaries.append("Error: Request timed out. Try reducing chunk size or increasing timeout.")
        except Exception as e:
            summaries.append(f"Error summarizing text: {e}")
        return summaries


def main():
    st.title("AI Research Paper Summarizer")
    st.write("Upload research papers to get concise, pointwise summaries.")

    study_bot = StudyBot()
    uploaded_files = st.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)

    if uploaded_files:
        with st.spinner("Processing uploaded files..."):
            temp_dir = tempfile.mkdtemp()
            documents = []

            for uploaded_file in uploaded_files:
                filepath = os.path.join(temp_dir, uploaded_file.name)
                with open(filepath, "wb") as f:
                    f.write(uploaded_file.read())
                text = study_bot.load_pdf(filepath)
                if text:
                    documents.append({"text": text})
                else:
                    st.error(f"Error processing file: {uploaded_file.name}")

            if documents:
                study_bot.initialize(documents)
                study_bot.generated_text = "\n".join([doc["text"] for doc in documents])

        st.success("Files uploaded and processed successfully!")

        if st.button("Summarize"):
            if study_bot.generated_text:
                with st.spinner("Generating summary..."):
                    summaries = study_bot.summarize(study_bot.generated_text)
                    st.subheader("Summary")
                    for idx, summary in enumerate(summaries, 1):
                        st.markdown(f"### Chunk {idx}")
                        st.write(summary)
            else:
                st.error("No documents to summarize.")

        if st.button("Clear Documents"):
            shutil.rmtree(temp_dir)
            st.success("Temporary files cleared.")


if __name__ == "__main__":
    main()