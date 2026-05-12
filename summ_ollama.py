import streamlit as st
import PyPDF2
from langchain_community.vectorstores import FAISS
from langchain.prompts import ChatPromptTemplate
from langchain_community.llms.ollama import Ollama

# Initialize the LLaMA model via Ollama
model_id = "llama3.1"
model = Ollama(model=model_id)

# Function to extract text from PDF
def extract_text_from_pdf(pdf_file):
    reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in range(len(reader.pages)):
        text += reader.pages[page].extract_text()
    return text

def summarize_text(input_text: str):
    """
    Summarize a given text using the llama3.1 model.

    Args:
        input_text (str): The text to summarize.

    Returns:
        summary (str): Summarized text.
    """
    # Step 1: Prepare the summarization prompt
    prompt_template = """
    You are an advanced summarization model. Summarize the following text:
    
    {text}
    
    Provide a concise and informative summary.
    """
    prompt = ChatPromptTemplate.from_template(prompt_template)

    # Step 2: Inject the whole document into the prompt
    input_prompt = prompt.format(text=input_text)

    # Step 3: Generate the summary
    result = model(input_prompt)
    
    return result.strip()

# Streamlit UI
st.title("PDF Summarizer with Llama3.1 Model")

uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

if uploaded_file is not None:
    # Extract text from uploaded PDF
    with st.spinner('Extracting text from PDF...'):
        document_text = extract_text_from_pdf(uploaded_file)

    # Summarize the extracted text
    if st.button("Summarize Text"):
        with st.spinner('Generating Summary...'):
            summary = summarize_text(document_text)
        
        st.subheader("Summary")
        st.markdown(summary)